import pandas as pd
from datasets import load_dataset


class CXMDataLoader:
    """
    Download and return the inputs for each task.
    Use load(task_key) with one of:
      "AQM", "KB_REFINEMENT", "ARTICLE_SEARCH",
      "INTENT_PREDICTION", "MULTI_TURN", "TOOL_CALLING"
    """

    @staticmethod
    def load_agent_quality_adherence() -> dict:
        ds = load_dataset("sprinklr-huggingface/CXM_Arena", "Agent_Quality_Adherence")
        return {"df": pd.DataFrame(ds["train"])}

    @staticmethod
    def load_article_refinement() -> dict:
        ds_kb = load_dataset("sprinklr-huggingface/CXM_Arena", "KB_Refinement")
        ds_articles = load_dataset("sprinklr-huggingface/CXM_Arena", "Articles")
        return {
            "similarity_df": pd.DataFrame(ds_kb["similarity_pairs"]),
            "contradictory_df": pd.DataFrame(ds_kb["contradictory_pairs"]),
            "articles_df": pd.DataFrame(ds_articles["KB_refinement_articles"]),
        }

    @staticmethod
    def load_article_search() -> dict:
        ds_q = load_dataset("sprinklr-huggingface/CXM_Arena", "Article_Search")
        ds_a = load_dataset("sprinklr-huggingface/CXM_Arena", "Articles")
        return {
            "questions_df": pd.DataFrame(ds_q["train"]),
            "articles_df": pd.DataFrame(ds_a["article_search_articles"]),
        }

    @staticmethod
    def load_intent_prediction() -> dict:
        ds_conv = load_dataset("sprinklr-huggingface/CXM_Arena", "Intent_Prediction")
        ds_tx = load_dataset("sprinklr-huggingface/CXM_Arena", "Taxonomy")
        return {
            "conversation_df": pd.DataFrame(ds_conv["train"]),
            "Taxonomy_1": pd.DataFrame(ds_tx["taxonomy_1"]),
            "Taxonomy_2": pd.DataFrame(ds_tx["taxonomy_2"]),
            "Taxonomy_3": pd.DataFrame(ds_tx["taxonomy_3"]),
        }

    @staticmethod
    def load_multi_turn_rag() -> dict:
        ds = load_dataset("sprinklr-huggingface/CXM_Arena", "Multi_Turn")
        ds_a = load_dataset("sprinklr-huggingface/CXM_Arena", "Articles")
        return {
            "conversation_df": pd.DataFrame(ds["train"]),
            "articles_df": pd.DataFrame(ds_a["multi_turn_articles"]),
        }

    @staticmethod
    def load_tool_calling() -> dict:
        ds = load_dataset("sprinklr-huggingface/CXM_Arena", "Tool_Calling")
        ds_a = load_dataset("sprinklr-huggingface/CXM_Arena", "Articles")
        ds_tools = load_dataset("sprinklr-huggingface/CXM_Arena", "Tools_Description")

        tools_dict = {}
        for _, row in pd.DataFrame(ds_tools['train']).iterrows():
            tools_dict[row.Name] = eval(row.Definition)

        return {
            "conversation_df": pd.DataFrame(ds["train"]),
            "articles_df": pd.DataFrame(ds_a["multi_turn_articles"]),
            "tools_dict": tools_dict,
        }

    @classmethod
    def load(cls, task_key: str) -> dict:
        key = task_key.upper()
        if key == "AQM":
            return cls.load_agent_quality_adherence()
        if key == "KB_REFINEMENT":
            return cls.load_article_refinement()
        if key == "ARTICLE_SEARCH":
            return cls.load_article_search()
        if key == "INTENT_PREDICTION":
            return cls.load_intent_prediction()
        if key == "MULTI_TURN_RAG":
            return cls.load_multi_turn_rag()
        if key == "TOOL_CALLING":
            return cls.load_tool_calling()
        raise KeyError(f"Unknown task_key {task_key!r} in DataLoader.load()")