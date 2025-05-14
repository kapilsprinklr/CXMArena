import random
import asyncio
from cxm_downloader import CXMDataLoader
from cxm_evaluator import CXMEvaluator
from cxm_predictor import CXMPredictor
from typing import Dict, Any, List, Tuple


class CXMTestSuite:
    """
    A comprehensive test suite for CXM Arena tasks.
    Each task has its own test method and uses the appropriate predictor and evaluator.
    """

    def __init__(self, test_size: int = 15):
        """
        Initialize the test suite with data loader, evaluator, and predictor.

        Args:
            test_size: Number of samples to use for testing each task
        """
        self.loader = CXMDataLoader()
        self.evaluator = CXMEvaluator()
        self.predictor = CXMPredictor()
        self.test_size = test_size

        # Define task configurations
        self.task_configs = {
            "AQM": {
                "model_name": "gemini-2.0-flash-001",
                "rps": 5
            },
            "KB_REFINEMENT": {
                "model_name": "intfloat/multilingual-e5-large-instruct",
                "k": 10,
                "similarity_threshold": 0.9,
            },
            "ARTICLE_SEARCH": {
                "model_name": "intfloat/multilingual-e5-large-instruct",
                "top_k": 10
            },
            "INTENT_PREDICTION": {
                "model_name": "gemini-2.0-flash-001",
                "rps": 5
            },
            "MULTI_TURN_RAG": {
                "model_name": "intfloat/multilingual-e5-large-instruct",
                "top_k": 10,
                "rps": 5
            },
            "TOOL_CALLING": {
                "model_name": "gemini-2.0-flash-001",
                "n_tools": [16, 32, 64, 96, 128],
                "rps": 5
            }
        }

    async def test_aqm(self) -> Tuple[float, float]:
        """Test Agent Quality Measurement task"""
        print("\nTesting AQM...")
        inp = self.loader.load("AQM")
        inp["df"] = inp["df"][:self.test_size]

        # Prepare questions list
        questions_list = [[qa['Question'] for qa in eval(js)] for js in inp["df"]["question_answers"].tolist()]

        # Get predictions
        predictions = await self.predictor.predict_aqm(
            inp["df"]["conversation"].tolist(),
            questions_list,
            model_name=self.task_configs["AQM"]["model_name"],
            rps=self.task_configs["AQM"]["rps"]
        )

        # Evaluate results
        result = await self.evaluator.evaluate("AQM", inp, predictions)
        print(f"AQM Results - Conversation Level Accuracy: {result[0]:.3f}, Question Level Accuracy: {result[1]:.3f}")
        return result

    async def test_kb_refinement(self) -> Dict[str, Tuple[float, float, float]]:
        """Test Knowledge Base Refinement task"""
        print("\nTesting KB Refinement...")
        inp = self.loader.load("KB_REFINEMENT")
        inp['articles_df'] = inp['articles_df'][:self.test_size]
        results = {}

        for pair_key in ["contradictory_df", "similarity_df"]:
            inp[pair_key] = inp[pair_key][:self.test_size]
            
            all_articles = list({article for pair in inp[pair_key]["Pairs"] for article in pair})
            predictions = self.predictor.predict_kb_refinement(inp, k = self.task_configs["KB_REFINEMENT"]["k"], similarity_threshold = self.task_configs["KB_REFINEMENT"]["similarity_threshold"], model_name = self.task_configs["KB_REFINEMENT"]["model_name"])

            # Evaluate results
            result = await self.evaluator.evaluate("KB_REFINEMENT", inp, predictions, pair_key=pair_key)
            results[pair_key] = result
            print(
                f"KB Refinement ({pair_key}) Results - Precision: {result[0]:.3f}, Recall: {result[1]:.3f}, F1: {result[2]:.3f}")

        return results

    async def test_article_search(self) -> float:
        """Test Article Search task"""
        print("\nTesting Article Search...")
        inp = self.loader.load("ARTICLE_SEARCH")
        inp["questions_df"] = inp["questions_df"][:self.test_size]

        # Get predictions
        predictions = self.predictor.predict_article_search(
            inp,
            top_k=self.task_configs["ARTICLE_SEARCH"]["top_k"],
            model_name=self.task_configs["ARTICLE_SEARCH"]["model_name"]
        )

        # Evaluate results
        result = await self.evaluator.evaluate("ARTICLE_SEARCH", inp, predictions)
        print(f"Article Search Results - Precision: {result:.3f}")
        return result

    async def test_intent_prediction(self) -> Dict[str, float]:
        """Test Intent Prediction task"""
        print("\nTesting Intent Prediction...")
        inp = self.loader.load("INTENT_PREDICTION")
        inp["conversation_df"] = inp["conversation_df"][:self.test_size]
        results = {}

        for taxonomy_level in ["Taxonomy_1", "Taxonomy_2", "Taxonomy_3"]:
            # Get predictions
            predictions = await self.predictor.predict_intent_fuzzy(
                inp,
                taxonomy_level=taxonomy_level,
                model_name=self.task_configs["INTENT_PREDICTION"]["model_name"],
                rps=self.task_configs["INTENT_PREDICTION"]["rps"]
            )

            # Evaluate results
            result = await self.evaluator.evaluate("INTENT_PREDICTION", inp, predictions, taxonomy_level=taxonomy_level)
            results[taxonomy_level] = result
            print(f"Intent Prediction ({taxonomy_level}) Results - Precision: {result:.3f}")

        return results

    async def test_multi_turn_rag(self) -> List[str]:
        """Test Multi-turn RAG task"""
        print("\nTesting Multi-turn RAG...")
        inp = self.loader.load("MULTI_TURN_RAG")
        inp["conversation_df"] = inp["conversation_df"][:self.test_size]

        # Get predictions
        kb_ids, answers = await self.predictor.predict_multi_turn_rag(
            inp,
            top_k=self.task_configs["MULTI_TURN_RAG"]["top_k"],
            model_name=self.task_configs["MULTI_TURN_RAG"]["model_name"],
            rps=self.task_configs["MULTI_TURN_RAG"]["rps"]
        )

        # Evaluate results
        result = await self.evaluator.evaluate_multi_turn_rag_results(
            inp,
            kb_ids,
            answers,
            model_name="gemini-1.5-pro"
        )
        print(f"Multi-turn RAG Results - Categories: {result}")
        return result

    async def test_tool_calling(self) -> Dict:
        """Test Tool Calling task"""
        print("\nTesting Tool Calling...")
        inp = self.loader.load("TOOL_CALLING")
        inp["conversation_df"] = inp["conversation_df"][:self.test_size]
        n_tools_list = self.task_configs["TOOL_CALLING"]["n_tools"]
        predictions = {}
        result = {}
        for n_tools in n_tools_list:
            predictions[n_tools] = await self.predictor.predict_tool_calling(
                inp,
                n_tools=n_tools,
                model_name=self.task_configs["TOOL_CALLING"]["model_name"],
                rps=self.task_configs["TOOL_CALLING"]["rps"]
            )

            # Evaluate results
            result[n_tools] = await self.evaluator.evaluate("TOOL_CALLING", inp, predictions[n_tools])
            print(f"Tool Calling Results (N_TOOLS:{n_tools}) - Precision: {result[n_tools]:.3f}")
        return result

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results"""
        results = {}

        # Run each test
        results["AQM"] = await self.test_aqm()
        results["KB_REFINEMENT"] = await self.test_kb_refinement()
        results["ARTICLE_SEARCH"] = await self.test_article_search()
        results["INTENT_PREDICTION"] = await self.test_intent_prediction()
        results["MULTI_TURN_RAG"] = await self.test_multi_turn_rag()
        results["TOOL_CALLING"] = await self.test_tool_calling()

        return results

def main():
    """Main function to run the test suite"""
    test_suite = CXMTestSuite(test_size=15)
    results = asyncio.run(test_suite.run_all_tests())

    # Print summary
    print("\nTest Suite Summary:")
    for task, result in results.items():
        print(f"\n{task}:")
        if isinstance(result, tuple):
            print(f"  Results: {result}")
        elif isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print(f"  Result: {result}")

if __name__ == "__main__":
    main()
