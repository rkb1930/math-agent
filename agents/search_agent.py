# agents/search_agent.py
import json
from typing import List, Dict, Any
import requests

import config


class MathWebSearchAgent:
    """Agent for searching web for math problems not in knowledge base"""

    def __init__(self, api_key: str = None):
        """
        Initialize the search agent

        Args:
            api_key: Tavily API key
        """
        self.api_key = api_key or config.TAVILY_API_KEY
        self.search_endpoint = "https://api.tavily.com/search"

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the web for math problems and solutions

        Args:
            query: Math problem to search for

        Returns:
            List of search results with content and sources
        """
        # Prepare query - focus on educational content
        enhanced_query = f"math problem solution step by step {query}"

        # Call Tavily API
        try:
            response = requests.post(
                self.search_endpoint,
                headers={"x-api-key": self.api_key},
                json={
                    "query": enhanced_query,
                    "search_depth": "advanced",
                    "include_answer": True,
                    "include_domains": [
                        "khanacademy.org",
                        "mathworld.wolfram.com",
                        "brilliant.org",
                        "purplemath.com",
                        "mathsisfun.com",
                        "stackoverflow.com",
                        "math.stackexchange.com"
                    ],
                    "max_results": config.MAX_SEARCH_RESULTS
                }
            )

            if response.status_code == 200:
                results = response.json()
                return self._process_search_results(results, query)
            else:
                print(f"Search API error: {response.status_code}")
                return []

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def _process_search_results(self, results: Dict[str, Any], original_query: str) -> List[Dict[str, Any]]:
        """
        Process and filter search results to extract math solutions

        Args:
            results: Raw search results
            original_query: Original math problem

        Returns:
            Processed search results
        """
        processed_results = []

        if "results" not in results:
            return processed_results

        for result in results.get("results", []):
            # Extract relevant information
            content = result.get("content", "")
            url = result.get("url", "")
            title = result.get("title", "")

            # Skip results that don't appear to have solutions
            if not any(keyword in content.lower() for keyword in [
                "solution", "step", "solve", "answer", "result", "approach"
            ]):
                continue

            processed_results.append({
                "content": content,
                "url": url,
                "title": title,
                "query": original_query
            })

        return processed_results