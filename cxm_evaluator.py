import ast
import json
import os
from typing import Any, Iterable, List, Sequence, Tuple, Union
from collections import defaultdict, Counter
from utils.vertex_ai_helper import VertexAIHelper
from utils.misc import get_kbs_content_string

import numpy as np
import pandas as pd

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def load_prompt(prompt_name):
    prompt_path = os.path.join(PROMPT_DIR, prompt_name)
    with open(prompt_path, "r") as f:
        return f.read()


class CXMEvaluator:
    """
    All metrics as methods.  Call evaluate(task_key, inp, results, **kwargs)
    with the same keys as DataLoader.load.
    """

    def __init__(self):
        self.task_prompt_files = {
            "MULTI_TURN_RAG": "multi_turn_rag_evaluation.txt",
            # Add more as needed
        }
        self.prompts = {k: load_prompt(v) for k, v in self.task_prompt_files.items()}
        self.llm = VertexAIHelper("./access_forrestor_nlp.json")

    @staticmethod
    def _safe_eval(obj: Any) -> Any:
        try:
            return ast.literal_eval(obj)
        except Exception:
            return obj

    @staticmethod
    def _parse_json(s, default_json=None):
        try:
            first = s.find('{')
            last = s.rfind('}')
            if first == -1 or last == -1 or first > last:
                raise ValueError("Input does not contain a valid JSON object based on '{' and '}'")
            json_str = s[first:last + 1]
            return json.loads(json_str)
        except Exception as e:
            print("Parsing JSON failed with error: {}".format(e))
            return default_json if default_json else None

    @staticmethod
    def _assert_length_match(n_true: int, n_pred: int, name: str) -> None:
        if n_true != n_pred:
            raise ValueError(f"Length mismatch for {name}: true={n_true}, pred={n_pred}")

    @staticmethod
    def _exact_match_precision(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
        if not y_true:
            return 0.0
        matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return matches / len(y_true)

    def aqm_conversation_level_accuracy(self, inp: dict, results: list):
        """
        Returns:
            - conversation_level_accuracy: proportion of conversations 100% correct
            - question_level_accuracy: overall proportion of questions correct
        """
        df = inp["df"]

        self._assert_length_match(len(df), len(results), "aqm results")

        # --- Extract answers as in your original code ---
        def extract(block):
            b = ast.literal_eval(block) if isinstance(block, str) else block
            if isinstance(b, list) and b and isinstance(b[0], dict):
                return [d["Answer"].lower() for d in b]
            if isinstance(b, list) and b and isinstance(b[0], str):
                return [s.lower() for s in b]
            raise TypeError("aqm_accuracy: expected list of dicts or strings")

        # Row-level true/pred
        true_flat, pred_flat = [], []
        for t_blk, p_blk in zip(df["question_answers"], results):
            true_flat.append(extract(t_blk))
            pred_flat.append(extract(p_blk))

        # Gather conversation_id mapping
        conversation_id_to_indices = defaultdict(list)
        for idx, cid in enumerate(df["conversation_id"]):
            conversation_id_to_indices[cid].append(idx)

        # --- Conversation-level accuracy ---
        n_conversations = len(conversation_id_to_indices)
        n_convo_correct = 0

        for cid, indices in conversation_id_to_indices.items():
            all_correct = True
            for idx in indices:
                if true_flat[idx] != pred_flat[idx]:
                    all_correct = False
                    break
            if all_correct:
                n_convo_correct += 1

        conversation_level_accuracy = n_convo_correct / n_conversations if n_conversations > 0 else 0.0

        # --- Question-level accuracy (flatten all arrays) ---
        q_total = 0
        q_correct = 0
        for tlist, plist in zip(true_flat, pred_flat):
            if len(tlist) != len(plist):
                # Mis-prediction of count, skip or count only where matching
                min_len = min(len(tlist), len(plist))
            else:
                min_len = len(tlist)
            for i in range(min_len):
                q_total += 1
                if tlist[i] == plist[i]:
                    q_correct += 1
        question_level_accuracy = q_correct / q_total if q_total > 0 else 0.0

        return conversation_level_accuracy, question_level_accuracy

    def article_refinement_metrics(
        self, inp: dict, results: Sequence, pair_key: str = "similarity_df"
    ) -> Tuple[float, float, float]:
        df = inp[pair_key]
        
        true_pairs = set()
        for x in df["Pairs"]:
            pair = self._safe_eval(x)
            true_pairs.add(tuple(sorted(pair)))
        
        pred_pairs = set()
        for x in results:
            pair = self._safe_eval(x)
            pred_pairs.add(tuple(sorted(pair)))

        if not pred_pairs:
            raise ValueError("No predictions provided for refinement metrics.")
        tp = len(true_pairs & pred_pairs)
        prec = tp / len(pred_pairs)
        rec = tp / len(true_pairs) if true_pairs else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    def article_search_precision(self, inp: dict, results: Sequence) -> float:
        return self._sequence_exact_match(inp["questions_df"], results, "True KB ID")

    def intent_exact_match_precision(
        self, inp: dict, results: Sequence, taxonomy_level: str = "Taxonomy_1"
    ) -> float:
        return self._sequence_exact_match(inp["conversation_df"], results, taxonomy_level)

    def rag_recall(self, inp: dict, results: Sequence[Iterable[Any]]) -> float:
        df = inp["conversation_df"]
        self._assert_length_match(len(df), len(results), "rag results")
        recalls = []
        for t_ids, p_ids in zip(df["kb_ids"], results):
            tset = set(self._safe_eval(t_ids))
            pset = set(self._safe_eval(p_ids))
            recalls.append((len(tset & pset) / len(tset)) if tset else 0.0)
        return float(np.mean(recalls))

    def tool_call_precision(self, inp: dict, results: Sequence) -> float:
        return self._sequence_exact_match(inp["conversation_df"], results, "true_tools")

    def _sequence_exact_match(
        self, df: pd.DataFrame, results: Sequence, truth_col: str
    ) -> float:
        if truth_col not in df.columns:
            raise KeyError(f"Column {truth_col!r} not found in DataFrame.")
        self._assert_length_match(len(df), len(results), truth_col)
        matches = 0
        for t, p in zip(df[truth_col], results):
            tv = self._safe_eval(t)
            pv = self._safe_eval(p)
            if isinstance(tv, list) and isinstance(pv, list):
                matches += sorted(tv) == sorted(pv)
            elif isinstance(pv,list):
                matches+= (tv in pv)
            else:
                matches += tv == pv
        return matches / len(df) if len(df) else 0.0

    async def evaluate_multi_turn_rag_results(self, multi_turn_rag_input, result_kbs_ids, result_answers, model_name='gemini-1.5-pro'):
        conversation_df = multi_turn_rag_input["conversation_df"]
        articles_df = multi_turn_rag_input["articles_df"]
        all_rag_classification_prompts = []

        i = 0
        multi_turn_rag_evaluation_template = self.prompts['MULTI_TURN_RAG']

        for _, row in conversation_df.iterrows():
            kb_id_true = self._safe_eval(row['kb_ids'])
            kb_id_pred = result_kbs_ids[i]
            kb_id_relevant = list(set(kb_id_pred) - set(kb_id_true))

            all_kbs_true = get_kbs_content_string(articles_df, kb_id_true)
            all_kbs_relevant = get_kbs_content_string(articles_df, kb_id_relevant)

            curr_prompt = multi_turn_rag_evaluation_template.format(
                conversation_context = row.conversation_context,
                llm_response = result_answers[i],
                true_kbs = all_kbs_true,
                relevant_kbs = all_kbs_relevant,
            )
            i += 1
            all_rag_classification_prompts.append(curr_prompt)

        classification_results = await self.llm.chat_batch(all_rag_classification_prompts, model_name=model_name)
        classification_results_parsed = [self._parse_json(x, default_json={'Category': 'parsing_error'})['Category'].lower() for x in classification_results]
        result =  dict(Counter(classification_results_parsed))
        result = {k: v/len(classification_results_parsed) for k, v in result.items()}
        return result

    async def evaluate(self, task_key: str, inp: dict, results, **kwargs):
        key = task_key.upper()
        if key == "AQM":
            return self.aqm_conversation_level_accuracy(inp, results)
        if key == "KB_REFINEMENT":
            assert 'pair_key' in kwargs and kwargs.get('pair_key') in ['contradictory_df', 'similarity_df']
            return self.article_refinement_metrics(inp, results, **kwargs)
        if key == "ARTICLE_SEARCH":
            return self.article_search_precision(inp, results)
        if key == "INTENT_PREDICTION":
            return self.intent_exact_match_precision(inp, results, **kwargs)
        if key == "MULTI_TURN_RAG":
            assert 'kb_ids' in results and 'answers' in results
            return await self.evaluate_multi_turn_rag_results(inp, results['kb_ids'], results['answers'], **kwargs)
        if key == "TOOL_CALLING":
            return self.tool_call_precision(inp, results)
        raise KeyError(f"Unknown task_key {task_key!r} in Evaluator.evaluate()")