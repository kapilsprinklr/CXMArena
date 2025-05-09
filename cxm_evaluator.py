import ast
from typing import Any, Iterable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

class CXMEvaluator:
    """
    All metrics as methods.  Call evaluate(task_key, inp, results, **kwargs)
    with the same keys as DataLoader.load.
    """

    @staticmethod
    def _safe_eval(obj: Any) -> Any:
        try:
            return ast.literal_eval(obj)
        except Exception:
            return obj

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

    def aqm_accuracy(self, inp: dict, results: Sequence) -> float:
        df = inp["df"]
        self._assert_length_match(len(df), len(results), "aqm results")

        def extract(block):
            b = self._safe_eval(block)
            if isinstance(b, list) and b and isinstance(b[0], dict):
                return [d["Answer"].lower() for d in b]
            if isinstance(b, list) and b and isinstance(b[0], str):
                return [s.lower() for s in b]
            raise TypeError("aqm_accuracy: expected list of dicts or strings")

        y_t, y_p = [], []
        for t_blk, p_blk in zip(df["question_answers"], results):
            y_t.extend(extract(t_blk))
            y_p.extend(extract(p_blk))
        return self._exact_match_precision(y_t, y_p)

    def article_denoising_metrics(
        self, inp: dict, results: Sequence, pair_key: str = "contradictory_df"
    ) -> Tuple[float, float, float]:
        df = inp[pair_key]
        true_pairs = {tuple(self._safe_eval(x)) for x in df["Pairs"]}
        pred_pairs = {tuple(self._safe_eval(x)) for x in results}
        if not pred_pairs:
            raise ValueError("No predictions provided for denoising metrics.")
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
            if isinstance(tv, list):
                matches += sorted(tv) == sorted(pv)
            else:
                matches += tv == pv
        return matches / len(df) if len(df) else 0.0

    def evaluate(self, task_key: str, inp: dict, results: Sequence, **kwargs):
        key = task_key.upper()
        if key == "AQM":
            return self.aqm_accuracy(inp, results)
        if key == "KB_DENOISING":
            return self.article_denoising_metrics(inp, results, **kwargs)
        if key == "ARTICLE_SEARCH":
            return self.article_search_precision(inp, results)
        if key == "INTENT_PREDICTION":
            return self.intent_exact_match_precision(inp, results, **kwargs)
        if key == "MULTI_TURN":
            return self.rag_recall(inp, results)
        if key == "TOOL_CALLING":
            return self.tool_call_precision(inp, results)
        raise KeyError(f"Unknown task_key {task_key!r} in Evaluator.evaluate()")