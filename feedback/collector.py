# feedback/collector.py
from typing import Dict, Any
import json
import os
from datetime import datetime


class FeedbackCollector:
    """Collects human feedback for math solutions"""

    def __init__(self, feedback_dir: str = "feedback_data"):
        """
        Initialize the feedback collector

        Args:
            feedback_dir: Directory to store feedback data
        """
        self.feedback_dir = feedback_dir
        os.makedirs(feedback_dir, exist_ok=True)

        # Load existing feedback if available
        self.feedback_file = os.path.join(feedback_dir, "feedback_log.jsonl")
        self.feedback_data = []
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, "r") as f:
                for line in f:
                    if line.strip():
                        self.feedback_data.append(json.loads(line))

    def collect_feedback(self, query: str, solution: str, path_taken: str,
                         ratings: Dict[str, int], comments: str = "") -> Dict[str, Any]:
        """
        Collect and store feedback for a solution

        Args:
            query: Original math problem
            solution: Solution provided
            path_taken: Solution strategy used (kb, search, direct)
            ratings: Dict of rating scores for different categories
            comments: Optional feedback comments

        Returns:
            Feedback entry
        """
        # Create feedback entry
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "solution_summary": solution[:100] + "..." if len(solution) > 100 else solution,
            "path_taken": path_taken,
            "ratings": ratings,
            "comments": comments,
            "average_rating": sum(ratings.values()) / len(ratings) if ratings else 0
        }

        # Add to memory and save to file
        self.feedback_data.append(feedback_entry)
        with open(self.feedback_file, "a") as f:
            f.write(json.dumps(feedback_entry) + "\n")

        return feedback_entry

    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Calculate statistics from collected feedback

        Returns:
            Statistics about the feedback
        """
        if not self.feedback_data:
            return {"total_feedback": 0}

        stats = {
            "total_feedback": len(self.feedback_data),
            "average_ratings": {},
            "path_performance": {}
        }

        # Calculate average ratings per category
        all_categories = set()
        for entry in self.feedback_data:
            all_categories.update(entry.get("ratings", {}).keys())

        for category in all_categories:
            ratings = [entry["ratings"].get(category, 0) for entry in self.feedback_data
                       if category in entry.get("ratings", {})]
            if ratings:
                stats["average_ratings"][category] = sum(ratings) / len(ratings)

        # Calculate performance by path taken
        paths = set(entry.get("path_taken") for entry in self.feedback_data)
        for path in paths:
            if not path:  # Skip None values
                continue

            path_entries = [entry for entry in self.feedback_data if entry.get("path_taken") == path]
            path_avg = sum(entry.get("average_rating", 0) for entry in path_entries) / len(path_entries)

            stats["path_performance"][path] = {
                "count": len(path_entries),
                "average_rating": path_avg
            }

        return stats
