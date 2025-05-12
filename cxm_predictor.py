import os
import time
from typing import List, Any
import ast

from call_vertex_ai import get_response

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def load_prompt(prompt_name):
    prompt_path = os.path.join(PROMPT_DIR, prompt_name)
    with open(prompt_path, "r") as f:
        return f.read()

class CXMPredictor:
    """
    Predict outputs for CXM Arena tasks using Gemini (Vertex AI).
    """
    def __init__(self, model="gemini-2.0-flash", max_retries=3):
        self.model = model
        self.max_retries = max_retries
        self.task_prompt_files = {
            "AQM": "aqm.txt",
            # Add more as needed
        }
        self.prompts = {k: load_prompt(v) for k,v in self.task_prompt_files.items()}


    
    
    def predict_aqm(self, conversations: List[str], question_lists: List[List[str]]) -> List[list]:
        def parse_yes_no_list(response_text: str) -> list:
            try:
                data = ast.literal_eval(response_text.strip("```").strip("json"))
                if isinstance(data, list):
                    return [str(qadict.get("Answer", "")).strip().lower() for qadict in data]
            except Exception as e:
                print(f"Parsing error:{response_text}", e)
            return []
        prompt_template = self.prompts["AQM"]
        preds = []
        for conv, qs in zip(conversations, question_lists):
            prompt = (prompt_template
                      .replace("<<Conversation>>", str(conv))
                      .replace("<<Questions>>", str(qs)))
            response = None
            for attempt in range(self.max_retries):
                try:
                    r = get_response([prompt], model=self.model)
                    if r and hasattr(r, "text"):
                        response = r.text
                        break
                except Exception as e:
                    print(f"Error: {e} (attempt {attempt+1})")
            preds.append(parse_yes_no_list(response) if response else [])
        return preds



    def predict(self, task_key: str, inp: dict) -> List[Any]:
        key = task_key.upper()
        if key == "AQM":
            df = inp["df"]
            questions_list=[[qa['Question'] for qa in ast.literal_eval(js)] for js in df["question_answers"].tolist()]
            return self.predict_aqm(df["conversation"].tolist(), questions_list)
        raise NotImplementedError(f"Task {task_key} not implemented in CXMPredictor.")
