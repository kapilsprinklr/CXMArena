import os
import time
from typing import List, Any
import ast
import asyncio

from utils.vertex_ai_helper import VertexAIHelper

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
        


    async def predict(self, task_key: str, inp: dict , model_name = "gemini-2.0-flash-001") -> List[Any]:
        key = task_key.upper()
        if key == "AQM":
            df = inp["df"]
            questions_list=[[qa['Question'] for qa in ast.literal_eval(js)] for js in df["question_answers"].tolist()]
            return await self.predict_aqm(df["conversation"].tolist(), questions_list,model_name,rps=6)
        raise NotImplementedError(f"Task {task_key} not implemented in CXMPredictor.")
