import os
import time
import json
from typing import List, Any, Dict, Union
import ast
import re
import asyncio
from tqdm.auto import tqdm
from utils.vertex_ai_helper import VertexAIHelper
from utils.misc import get_kbs_content_string
from thefuzz import process
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from collections import OrderedDict
import numpy as np


PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def load_prompt(prompt_name):
    prompt_path = os.path.join(PROMPT_DIR, prompt_name)
    with open(prompt_path, "r") as f:
        return f.read()

class CXMPredictor:
    """
    Predict outputs for CXM Arena tasks using Gemini (Vertex AI).
    """
    def __init__(self):
        self.task_prompt_files = {
            "AQM": "aqm.txt",
            "INTENT_PREDICTION": "intent_prediction.txt",
            "QUERY_FORMATION": "query_formation.txt",
            "TOOL_CALLING": "tool_calling.txt",
            "MULTI_TURN_RAG_RESPONSE": "multi_turn_agent_response.txt",
            # Add more as needed
        }
        self.prompts = {k: load_prompt(v) for k,v in self.task_prompt_files.items()}
        self.llm = VertexAIHelper("./access_forrestor_nlp.json")
    
    @staticmethod
    def _parse_conversation(conversation_text: str) -> List[tuple]:
        """
        Parse a conversation string like "Customer: ... Agent: ..." into a list of
        (role, text) tuples where role is 'user' for Customer and 'model' for Agent.
        """
        pattern = r"(Customer|Agent): ([\s\S]*?)(?=(?:Customer|Agent): |$)"
        return [
            ('user' if m.group(1) == 'Customer' else 'model', m.group(2).strip())
            for m in re.finditer(pattern, conversation_text)
        ]
    
    @staticmethod
    def _retrieve_top_k(
        documents: List[str],
        document_ids: List[str],
        queries: List[str],
        top_k: int,
        chunk_size: int,
        chunk_overlap: int,
        embed_model_name: str
    ) -> List[List[str]]:
        """
        Split documents into chunks, embed with SentenceTransformer, index with FAISS,
        and for each query retrieve the top_k unique document_ids by cosine similarity.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = []
        chunk_metadata = []
        for doc_id, doc in zip(document_ids, documents):
            doc_chunks = text_splitter.split_text(doc)
            chunks.extend(doc_chunks)
            chunk_metadata.extend([doc_id] * len(doc_chunks))

        embedding_model = SentenceTransformer(embed_model_name)
        # Encode and normalize
        chunk_embeddings = embedding_model.encode(
            chunks, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True
        ).cpu().numpy()
        query_embeddings = embedding_model.encode(
            queries, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True
        ).cpu().numpy()

        dim = chunk_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(chunk_embeddings)
        distances, indices = index.search(query_embeddings, top_k)

        results = []
        for idx_list in indices:
            retrieved_ids = [chunk_metadata[i] for i in idx_list]
            unique_ids = list(OrderedDict.fromkeys(retrieved_ids))[:top_k]
            results.append(unique_ids)
        return results

    async def predict_aqm(
        self,
        conversations: List[str],
        question_lists: List[List[str]],
        model_name: str,
        rps: int = 5,          # Maximum number of requests to start per second
        max_concurrent: int = 10 # Maximum in-flight requests at any one time (optional, defaults to rps*2)
    ) -> List[list]:
        def parse_yes_no_list(response_text: str) -> list:
            try:
                data = ast.literal_eval(response_text.strip("```").strip("json"))
                if isinstance(data, list):
                    return [str(qadict.get("Answer", "")).strip().lower() for qadict in data]
            except Exception as e:
                print(f"Parsing error:{response_text}", e)
            return []

        prompt_template = self.prompts["AQM"]
        prompts = []

        # Create all prompts
        for conv, qs in zip(conversations, question_lists):
            prompt = (
                prompt_template
                .replace("<<Conversation>>", str(conv))
                .replace("<<Questions>>", str(qs))
            )
            prompts.append([("user", prompt)])

        # Use the batch method which already handles RPS and concurrency
        responses = await self.llm.chat_batch(
            all_messages=prompts,
            model_name=model_name,
            rps=rps,
            max_concurrent=max_concurrent
        )

        # Parse the responses
        predictions = [parse_yes_no_list(resp) if resp else [] for resp in responses]
        predictions = [pred if pred else ['parsing_error' for _ in range(len(qs))] for pred, qs in zip(predictions, question_lists)]
        return predictions

    async def predict_intent_fuzzy(
        self,
        inp: dict,
        taxonomy_level: str = "Taxonomy_1",
        model_name: str = "gemini-2.0-flash-001",
        rps: int = 6,
        prompt_file: str = "contact_driver.txt",
    ) -> list:
        """
        Returns:
            List of predicted intent names, fuzzily matched to those from the provided taxonomy_level.
        """
        # Prepare taxonomy id list
        taxonomy_df = inp[taxonomy_level]
        if taxonomy_level in ["Taxonomy_1", "Taxonomy_3"]:
            candidates = [str(row["L1"]).strip() for _, row in taxonomy_df.iterrows()]
            taxonomy_dict = {str(row["L1"]).strip(): str(row["Description"]).strip() for _, row in taxonomy_df.iterrows()}
        elif taxonomy_level == "Taxonomy_2":
            candidates = [f"{str(row['L1']).strip()}:{str(row['L2']).strip()}" for _, row in taxonomy_df.iterrows()]
            taxonomy_dict = {f"{str(row['L1']).strip()}:{str(row['L2']).strip()}": str(row["Description"]).strip() for _, row in taxonomy_df.iterrows()}
        taxonomy_str = "\n".join([f"{i+1}. {k}: {v}" for i, (k, v) in enumerate(taxonomy_dict.items())])

        # Load prompt template (as defined in your prompt_dir)
        prompt_template = self.prompts.get("INTENT_PREDICTION")
        if not prompt_template:
            prompt_template = load_prompt(prompt_file)
            self.prompts["INTENT_PREDICTION"] = prompt_template

        inp_df = inp["conversation_df"]
        preds = [None] * len(inp_df)
        semaphore = asyncio.Semaphore(rps*2)

        async def call_one(i, summary):
            prompt = prompt_template.format(summary, taxonomy_str)
            async with semaphore:
                response = await self.llm.chat([("user", prompt)], model_name=model_name)
                clean = str(response).strip().strip("```").strip("json")
                try:
                    js = json.loads(clean)
                    intent_raw = js.get('Intent','')
                except Exception:
                    try:
                        js = ast.literal_eval(clean)
                        intent_raw = js.get('Intent','')
                    except Exception:
                        intent_raw = "others"
                # FUZZY MATCH
                intent_raw = str(intent_raw).strip()
                match = process.extractOne(intent_raw, [c for c in candidates], score_cutoff=70)
                mapped_pred = match[0] if match else "others"
                preds[i] = mapped_pred

        tasks = []
        for i, row in tqdm(inp_df.iterrows(),total=inp_df.shape[0]):
            summary = row["Complete Conversations"]
            tasks.append(call_one(i, summary))
            if (i+1)%rps==0:
                await asyncio.gather(*tasks[-rps:]);
        remain = len(inp_df)%rps
        if remain: await asyncio.gather(*tasks[-remain:])
        return preds


    def predict_article_search(
        self, inp: dict, top_k=10, chunk_size=1000, chunk_overlap=100,model_name = "intfloat/multilingual-e5-large-instruct"
    ) -> list:
        """
        For each query, return the top_k document_ids (str) from articles_df.
        """
        questions_df = inp["questions_df"]
        articles_df = inp["articles_df"]

        queries = questions_df["query"].tolist()
        article_ids = articles_df["document_id"].astype(str).tolist()
        articles = articles_df["document_content"].tolist()

        return self._retrieve_top_k(
            documents=articles,
            document_ids=article_ids,
            queries=queries,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_model_name=model_name,
        )

    async def _generate_messages(self, conversations, list_of_kb_ids, articles_df, rps):
        list_of_messages = []
        list_of_system_prompts = []
        system_prompt_template = self.prompts['MULTI_TURN_RAG_RESPONSE']

        for conv_text, kb_ids in zip(conversations, list_of_kb_ids):
            curr_messages = self._parse_conversation(conv_text)
            curr_system_prompt = system_prompt_template.format(
                kb_content=get_kbs_content_string(articles_df, kb_ids)
            )

            list_of_messages.append(curr_messages)
            list_of_system_prompts.append(curr_system_prompt)

        results = await self.llm.chat_batch(
            all_messages=list_of_messages,
            system_message=list_of_system_prompts,
            model_name='gemini-2.0-flash',
            rps=rps
        )

        return results


    async def predict_multi_turn_rag(
            self, inp: dict, top_k=10, chunk_size=1000, chunk_overlap=100,
            model_name="intfloat/multilingual-e5-large-instruct", rps = 1.0
    ) -> list:
        """
        For each query, return the top_k document_ids (str) from articles_df.
        """
        conversation_df = inp["conversation_df"]
        articles_df = inp["articles_df"]

        conversations = conversation_df['conversation_context'].tolist()

        prompt_template = self.prompts['QUERY_FORMATION']
        prompts = [prompt_template.format(conversation=conv) for conv in conversations]
        queries = await self.llm.chat_batch(prompts, model_name='gemini-2.0-flash')
        article_ids = articles_df["document_id"].astype(str).tolist()
        articles = articles_df["document_content"].tolist()

        results_kb_ids = self._retrieve_top_k(
            documents=articles,
            document_ids=article_ids,
            queries=queries,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_model_name=model_name,
        )

        agent_responses = await self._generate_messages(conversations, results_kb_ids, articles_df, rps)
        return results_kb_ids, agent_responses

    async def predict_tool_calling(
        self,
        inp: dict,
        n_tools: int = 16,
        model_name: str = "gemini-2.0-flash-001",
        rps: float = 1.0,
        # prompt_file: str = "tool_calling.txt",
    ) -> list:

        conv_df = inp['conversation_df']
        articles_df = inp['articles_df']
        tools_dict = inp['tools_dict']
        system_prompt_template = self.prompts["TOOL_CALLING"]

        list_of_messages = []
        list_of_system_prompts = []
        list_of_tools = []

        for _, row in conv_df.iterrows():
            curr_messages = self._parse_conversation(row.conversation_context)
            curr_tools = row[f"tool_candidates{n_tools}"]
            curr_tools_openai_format = [tools_dict[x] for x in curr_tools]
            curr_system_prompt = system_prompt_template.format(
                kb_content = get_kbs_content_string(articles_df, row.kb_ids)
            )

            list_of_messages.append(curr_messages)
            list_of_system_prompts.append(curr_system_prompt)
            list_of_tools.append(curr_tools_openai_format)

        results = await self.llm.chat_batch(
            all_messages=list_of_messages,
            system_message=list_of_system_prompts,
            tools=list_of_tools,
            model_name=model_name,
            rps=rps
        )

        processed_results = []
        for x in results:
            if isinstance(x, str) or not x:  # Handle string or empty list
                processed_results.append(None)
                continue
                
            try:
                tool_names = [y['name'] for y in x if isinstance(y, dict) and 'name' in y]
                processed_results.append(list(set(tool_names)) if tool_names else None)
            except:
                print("parsing failed for tool call result:", x)
                processed_results.append(None)
                
        results = processed_results
        return results

    def predict_kb_refinement(self, inp: dict, similarity_threshold: float = 0.9, k = 5, model_name: str = "intfloat/multilingual-e5-large-instruct") -> list[list[str]]:
        """
       Predicts similar pairs of articles for knowledge base refinement based on semantic similarity.
    
        Args:
        inp (dict): Input dictionary containing 'articles_df' with document IDs and content.
        similarity_threshold (float, optional): Threshold for cosine similarity to consider articles as similar.
            Defaults to 0.8.
        k (int, optional): Number of nearest neighbors to retrieve for each article. Defaults to 5.
        model_name (str, optional): Name of the sentence transformer model to use for embeddings.
            Defaults to "intfloat/multilingual-e5-large-instruct".
    
        Returns:
        list[list[str]]: List of similar article pairs where each pair is a list of two string IDs.
            Format: [[id1, id2], [id3, id4], ...]. Each pair represents articles with 
            similarity above the threshold. The list may be empty if no similar pairs are found.
        
        """
        articles_df = inp['articles_df']
        article_ids = articles_df["document_id"].astype(str).tolist()
        articles = articles_df["document_content"].tolist()
        
        embedding_model = SentenceTransformer(model_name)

        article_embeddings = embedding_model.encode(
            articles, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True).cpu().numpy()

        article_embeddings = np.array(article_embeddings).astype('float32')
        index = faiss.IndexFlatIP(article_embeddings.shape[1])
        index.add(article_embeddings)

        similarities, indices = index.search(article_embeddings, k)
        
        similar_pairs = []
        for i in range(indices.shape[0]):
            for j, sim in zip(indices[i], similarities[i]):
                if i < j and sim >= similarity_threshold:  
                    similar_pairs.append([article_ids[i], article_ids[j]])

        return similar_pairs


    async def predict(self, task_key: str, inp: dict , model_name = "gemini-2.0-flash-001") -> Union[List[Any], Dict]:
        key = task_key.upper()
        if key == "AQM":
            df = inp["df"]
            questions_list=[[qa['Question'] for qa in ast.literal_eval(js)] for js in df["question_answers"].tolist()]
            return await self.predict_aqm(df["conversation"].tolist(), questions_list,model_name,rps=6)
        elif key == "INTENT_PREDICTION":
            predictions = {}
            for i in range(1,4):
                predictions[f"Taxonomy_{i}"]=await self.predict_intent_fuzzy(inp,taxonomy_level=f"Taxonomy_{i}",rps=20)
            return predictions
        elif key == "ARTICLE_SEARCH":
            return self.predict_article_search(inp, top_k=10,model_name="intfloat/multilingual-e5-large-instruct")
        elif key == "MULTI_TURN_RAG":
            return await self.predict_multi_turn_rag(
                inp, top_k = 10, chunk_size = 1000, chunk_overlap = 100, model_name = "intfloat/multilingual-e5-large-instruct", rps = 1.0
            )
        elif key == "TOOL_CALLING":
            return await self.predict_tool_calling(inp, n_tools = 16, model_name = "gemini-2.0-flash-001", rps = 1.0)
        elif key == "KB_REFINEMENT":
            return self.predict_kb_refinement(inp, similarity_threshold=0.9, k = 5, model_name="intfloat/multilingual-e5-large-instruct")

        raise NotImplementedError(f"Task {task_key} not implemented in CXMPredictor.")