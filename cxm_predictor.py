import os
import time
import json
from typing import List, Any
import ast
import re
import asyncio
from tqdm.auto import tqdm
from utils.vertex_ai_helper import VertexAIHelper
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
            "CONTACT_DRIVER": "contact_driver.txt",
            "QUERY_FORMATION": "query_formation.txt",
            "TOOL_CALLING": "tool_calling.txt",
            # Add more as needed
        }
        self.prompts = {k: load_prompt(v) for k,v in self.task_prompt_files.items()}
        self.llm = VertexAIHelper("./access_forrestor_nlp.json")




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
        n = len(conversations)
        preds = [None] * n

        # RPS throttle and concurrency semaphore
        max_concurrent = max_concurrent or rps * 2
        semaphore = asyncio.Semaphore(max_concurrent)

        async def call_one(i, conv, qs):
            prompt = (
                prompt_template
                .replace("<<Conversation>>", str(conv))
                .replace("<<Questions>>", str(qs))
            )
            async with semaphore:
                response = await self.llm.chat(messages=[("user", prompt)], model_name=model_name)
                preds[i] = parse_yes_no_list(response) if response else []

        # Fire tasks in waves of 'rps'
        tasks = []
        for idx, (conv, qs) in enumerate(zip(conversations, question_lists)):
            tasks.append(call_one(idx, conv, qs))
            if (idx + 1) % rps == 0:
                await asyncio.gather(*tasks[-rps:])  # Wait for last batch to finish
                await asyncio.sleep(1)               # Wait 1s to enforce RPS

        # Finish any remaining tasks
        if (n % rps) != 0:
            await asyncio.gather(*tasks[-(n % rps):])
        # preds may fill out of order, but is guaranteed to be fully filled since each call_one writes by index

        return preds

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
        prompt_template = self.prompts.get("CONTACT_DRIVER")
        if not prompt_template:
            prompt_template = load_prompt(prompt_file)
            self.prompts["CONTACT_DRIVER"] = prompt_template

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

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
        )

        chunks = []
        chunk_metadata = []
        for article_id, article in zip(article_ids, articles):
            article_chunks = text_splitter.split_text(article)
            chunks.extend(article_chunks)
            chunk_metadata.extend([article_id] * len(article_chunks))

        # Load embedding model (cached for this instance)
        embedding_model = SentenceTransformer(model_name)

        chunk_embeddings = embedding_model.encode(
            chunks, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True
        ).cpu().numpy()

        query_embeddings = embedding_model.encode(
            queries, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True
        ).cpu().numpy()

        dimension = chunk_embeddings.shape[1]
        print(f"Indexing Articles")

        # FAISS: inner product (assumes normalized vectors = cosine similarity)
        index = faiss.IndexFlatIP(dimension)
        index.add(chunk_embeddings)
        distances, indices = index.search(query_embeddings, top_k)

        print(f"Fetching Articles")

        # For each query, retreive unique article_ids by chunk-sim order
        results = []
        for chunk_idxs in indices:
            retrieved_article_ids = [chunk_metadata[idx] for idx in chunk_idxs]
            unique_article_ids = list(OrderedDict.fromkeys(retrieved_article_ids))[:top_k]
            results.append(unique_article_ids)
        return results

    async def predict_multi_turn_article_search(
            self, inp: dict, top_k=10, chunk_size=1000, chunk_overlap=100,
            model_name="intfloat/multilingual-e5-large-instruct"
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

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
        )

        chunks = []
        chunk_metadata = []
        for article_id, article in zip(article_ids, articles):
            article_chunks = text_splitter.split_text(article)
            chunks.extend(article_chunks)
            chunk_metadata.extend([article_id] * len(article_chunks))

        # Load embedding model (cached for this instance)
        embedding_model = SentenceTransformer(model_name)

        chunk_embeddings = embedding_model.encode(
            chunks, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True
        ).cpu().numpy()

        query_embeddings = embedding_model.encode(
            queries, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True
        ).cpu().numpy()

        dimension = chunk_embeddings.shape[1]
        print(f"Indexing Articles")

        # FAISS: inner product (assumes normalized vectors = cosine similarity)
        index = faiss.IndexFlatIP(dimension)
        index.add(chunk_embeddings)
        distances, indices = index.search(query_embeddings, top_k)

        print(f"Fetching Articles")

        # For each query, retreive unique article_ids by chunk-sim order
        results = []
        for chunk_idxs in indices:
            retrieved_article_ids = [chunk_metadata[idx] for idx in chunk_idxs]
            unique_article_ids = list(OrderedDict.fromkeys(retrieved_article_ids))[:top_k]
            results.append(unique_article_ids)
        return results

    def get_kb_content(self, articles_df, kb_id):
        rows = articles_df.loc[articles_df.document_id == kb_id]
        return rows.document_content.tolist()[0]

    def get_kbs_content_string(self, articles_df, kb_ids):
        all_kbs = [f"KB {i+1} ID: {kb_id}\n KB {i+1} Content: {self.get_kb_content(articles_df, kb_id)}" for i, kb_id in enumerate(kb_ids)]
        line_br = '-'*50
        return f"\n{line_br}\n".join(all_kbs)

    async def predict_tool_calling(
        self,
        inp: dict,
        n_tools: int = 16,
        model_name: str = "gemini-2.0-flash-001",
        rps: int = 6,
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
            conversation_text = row.conversation_context
            pattern = r"(Customer|Agent): ([\s\S]*?)(?=(?:Customer|Agent): |$)"
            curr_messages = [
                ('user' if match.group(1) == 'Customer' else 'model', match.group(2).strip())
                for match in re.finditer(pattern, conversation_text)
            ]
            curr_tools = row[f"tool_candidates{n_tools}"]
            curr_tools_openai_format = [tools_dict[x] for x in curr_tools]
            curr_system_prompt = system_prompt_template.format(
                kb_content = self.get_kbs_content_string(articles_df, row.kb_ids)
            )

            list_of_messages.append(curr_messages)
            list_of_system_prompts.append(curr_system_prompt)
            list_of_tools.append(curr_tools_openai_format)

        results = await self.llm.chat_batch(
            all_messages=list_of_messages,
            system_message=list_of_system_prompts,
            tools=list_of_tools,
            model_name=model_name
        )

        return results


    async def predict(self, task_key: str, inp: dict , model_name = "gemini-2.0-flash-001") -> List[Any]:
        key = task_key.upper()
        if key == "AQM":
            df = inp["df"]
            questions_list=[[qa['Question'] for qa in ast.literal_eval(js)] for js in df["question_answers"].tolist()]
            return await self.predict_aqm(df["conversation"].tolist(), questions_list,model_name,rps=6)
        elif key == "CONTACT_DRIVER":
            predictions = {}
            for i in range(1,4):
                predictions[f"Taxonomy_{i}"]=await self.predict_intent_fuzzy(inp,taxonomy_level=f"Taxonomy_{i}",rps=20)
            return predictions
        elif key == "ARTICLE_SEARCH":
            return self.predict_article_search(inp, top_k=10,model_name="intfloat/multilingual-e5-large-instruct")

        raise NotImplementedError(f"Task {task_key} not implemented in CXMPredictor.")
