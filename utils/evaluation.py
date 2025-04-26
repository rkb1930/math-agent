# utils/evaluation.py
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, Any
import requests

from agents.router import MathAgentRouter


class JEEBenchEvaluator:
    """Evaluates the Math Agent against JEE benchmark questions"""

    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize the evaluator

        Args:
            output_dir: Directory to store evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.router = MathAgentRouter()

    def download_jee_benchmark(self):
        """Download JEE benchmark dataset if not available"""
        dataset_path = os.path.join(self.output_dir, "jee_benchmark.csv")

        if os.path.exists(dataset_path):
            return dataset_path

        # Using a small sample of JEE questions from GitHub
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.csv"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(dataset_path, "w") as f:
                    f.write(response.text)
                print(f"Downloaded JEE benchmark to {dataset_path}")
                return dataset_path
            else:
                print(f"Failed to download dataset: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None

    def run_evaluation(self, num_samples: int = 20) -> Dict[str, Any]:
        """
        Run evaluation on JEE benchmark

        Args:
            num_samples: Number of questions to evaluate

        Returns:
            Evaluation metrics
        """
        dataset_path = self.download_jee_benchmark()
        if not dataset_path:
            return {"success": False, "error": "Could not load benchmark dataset"}

        # Load dataset
        df = pd.read_csv(dataset_path)
        if len(df) > num_samples:
            df = df.sample(num_samples, random_state=42)

        results = []
        routes_taken = {"knowledge_base": 0, "web_search": 0, "direct_solution": 0}

        # Run evaluation on each problem
        for i, row in df.iterrows():
            problem = row.get("question", "")
            expected_answer = row.get("answer", "")

            if not problem or not expected_answer:
                continue

            print(f"Evaluating problem {i + 1}/{len(df)}")

            # Get solution from our system
            solution, metadata = self.router.process_query(problem)

            # Track routes taken
            route = metadata.get("path_taken", "direct_solution")
            routes_taken[route] = routes_taken.get(route, 0) + 1

            # Evaluate the solution (simplified)
            result = {
                "problem": problem,
                "expected": expected_answer,
                "solution": solution,
                "route": route,
                "confidence": metadata.get("confidence", 0)
            }
            results.append(result)

        # Calculate metrics (basic evaluation)
        metrics = {
            "total_problems": len(results),
            "routes_taken": routes_taken,
            "avg_confidence": np.mean([r["confidence"] for r in results]) if results else 0
        }

        # Save results
        results_path = os.path.join(self.output_dir, "jee_evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)

        print(f"Evaluation complete. Results saved to {results_path}")
        return metrics
