import random
from cxm_downloader import CXMDataLoader
from cxm_evaluator import CXMEvaluator
from cxm_predictor import CXMPredictor
import asyncio
import pickle as pkl

# Initialize loader and evaluator
cxm_loader = CXMDataLoader()
cxm_evaluator = CXMEvaluator()

# 1) Agent Quality Measurement (AQM)
aqm_input = cxm_loader.load("AQM")
aqm_dataframe = aqm_input["df"]
aqm_random_predictions = [random.choices(["yes", "no"], k=len(eval(answers))) for answers in aqm_dataframe["question_answers"]]
print("Random AQM:", cxm_evaluator.evaluate("AQM", aqm_input, aqm_random_predictions))

aqm_input2 = {"df":aqm_dataframe[:12]}
print("Running AQM Predictor")
predictor = CXMPredictor()
predictions = asyncio.run(predictor.predict("AQM",aqm_input2,"gemini-2.0-flash-001"))
print(f"Sample predictions are {predictions[0]}")
print("Actual AQM using 10 inputs:", cxm_evaluator.evaluate("AQM", aqm_input2, predictions))


# 2) Knowledge Base Denoising (KB_DENOISING)
kb_denoising_input = cxm_loader.load("KB_DENOISING")
contradictory_pairs_df = kb_denoising_input["contradictory_df"]
all_articles = list({article for pair in contradictory_pairs_df["Pairs"] for article in pair})
kb_denoising_random_predictions = [random.choices(all_articles, k=2) for _ in range(len(contradictory_pairs_df))]
print("KB_DENOISING P/R/F1:", cxm_evaluator.evaluate("KB_DENOISING", kb_denoising_input, kb_denoising_random_predictions))

similarity_pairs_df = kb_denoising_input["similarity_df"]
all_articles_similarity = list({article for pair in contradictory_pairs_df["Pairs"] for article in pair})
similarity_random_predictions = [random.choices(all_articles_similarity, k=2) for _ in range(len(contradictory_pairs_df))]
print("KB_DENOISING P/R/F1:", cxm_evaluator.evaluate("KB_DENOISING", kb_denoising_input, similarity_random_predictions))

# 3) Article Search
article_search_input = cxm_loader.load("ARTICLE_SEARCH")
questions_df = article_search_input["questions_df"]
article_search_predictions = list(questions_df["True KB ID"])
random.shuffle(article_search_predictions)
print("ARTICLE_SEARCH P@1:", cxm_evaluator.evaluate("ARTICLE_SEARCH", article_search_input, article_search_predictions))

# 4) Intent Prediction
intent_prediction_input = cxm_loader.load("INTENT_PREDICTION")
intent_conversation_df = intent_prediction_input["conversation_df"]
taxonomy_choices = intent_prediction_input["Taxonomy_1"]["L1"]
intent_random_predictions = random.choices(taxonomy_choices, k=len(intent_conversation_df))
print("INTENT_PREDICTION EM:",
      cxm_evaluator.evaluate("INTENT_PREDICTION", intent_prediction_input, intent_random_predictions, taxonomy_level="Taxonomy_1"))

# 5) Multi-Turn RAG
multi_turn_input = cxm_loader.load("MULTI_TURN")
multi_turn_conversation_df = multi_turn_input["conversation_df"]
all_kb_ids = [kb_id for row in multi_turn_conversation_df["kb_ids"] for kb_id in eval(row)]
multi_turn_random_predictions = [random.sample(all_kb_ids, k=10) for _ in range(len(multi_turn_conversation_df))]
print("MULTI_TURN RAG Recall:", cxm_evaluator.evaluate("MULTI_TURN", multi_turn_input, multi_turn_random_predictions))

# 6) Tool Calling
tool_calling_input = cxm_loader.load("TOOL_CALLING")
tool_calling_conversation_df = tool_calling_input["conversation_df"]
all_tools = [tool for tool_list in tool_calling_conversation_df["true_tools"] for tool in tool_list]
tool_calling_random_predictions = [[random.choice(all_tools)] for _ in range(len(tool_calling_conversation_df))]
print("TOOL_CALLING Precision:", cxm_evaluator.evaluate("TOOL_CALLING", tool_calling_input, tool_calling_random_predictions))