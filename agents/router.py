# agents/router.py
from typing import Dict, Any, List, Tuple
import json

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from data.vectordb import MathKnowledgeBase
from agents.search_agent import MathWebSearchAgent
from agents.solution_agent import MathSolutionAgent
from utils.guardrails import MathGuardrails

import config


class MathAgentRouter:
    """Main router for the Math Agent system"""

    def __init__(self):
        """Initialize the router and all components"""
        # Initialize components
        self.knowledge_base = MathKnowledgeBase()
        self.search_agent = MathWebSearchAgent()
        self.solution_agent = MathSolutionAgent()
        self.guardrails = MathGuardrails()

        # Initialize LLM for routing decisions
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.1
        )

        # Prompt for categorizing math problems
        self.categorize_prompt = PromptTemplate(
            input_variables=["problem"],
            template=""" 
            You are a math professor who needs to categorize a math problem.

            PROBLEM:
            {problem}

            Please categorize this problem into ONE of these categories:
            - algebra
            - calculus
            - geometry
            - probability
            - statistics
            - number_theory
            - linear_algebra
            - differential_equations
            - trigonometry
            - other

            Respond with just the category name, nothing else.
            """
        )

        self.categorize_chain = LLMChain(llm=self.llm, prompt=self.categorize_prompt)

        # Tracking for feedback
        self.query_log = {}

    def process_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a math query through the entire system

        Args:
            query: User's math question

        Returns:
            Tuple of (solution, metadata)
        """
        # Track query for feedback
        query_id = hash(query)
        self.query_log[query_id] = {
            "query": query,
            "path_taken": None,
            "solution": None,
            "feedback": None
        }

        # Apply input guardrails
        valid, message = self.guardrails.validate_input(query)
        if not valid:
            return message, {"success": False, "reason": message}

        # Categorize the math problem
        category = self._categorize_problem(query)

        # Try knowledge base first
        kb_results = self.knowledge_base.search(query, top_k=3)

        solution = ""
        metadata = {
            "success": True,
            "query": query,
            "category": category,
            "path_taken": None,
            "confidence": 0.0,
            "sources": []
        }

        # Check if knowledge base has relevant results
        if kb_results and any(result["score"] >= config.SIMILARITY_THRESHOLD for result in kb_results):
            solution = self.solution_agent.generate_from_knowledge_base(query, kb_results)
            metadata["path_taken"] = "knowledge_base"
            metadata["confidence"] = max(result["score"] for result in kb_results)
            metadata["sources"] = [
                {"type": "knowledge_base", "problem": result["problem"][:100] + "..."}
                for result in kb_results
            ]
        else:
            # Try web search
            search_results = self.search_agent.search(query)

            if search_results:
                solution = self.solution_agent.generate_from_search(query, search_results)
                metadata["path_taken"] = "web_search"
                metadata["confidence"] = 0.7  # Approximate confidence for web results
                metadata["sources"] = [
                    {"type": "web", "title": result["title"], "url": result["url"]}
                    for result in search_results
                ]
            else:
                # Fall back to direct solution
                solution = self.solution_agent.generate_direct_solution(query)
                metadata["path_taken"] = "direct_solution"
                metadata["confidence"] = 0.5  # Lower confidence for direct solutions
                metadata["sources"] = [{"type": "model_knowledge", "name": config.LLM_MODEL}]

        # Apply output guardrails
        solution = self.guardrails.sanitize_output(solution)

        # Update query log
        self.query_log[query_id]["path_taken"] = metadata["path_taken"]
        self.query_log[query_id]["solution"] = solution

        return solution, metadata

    def _categorize_problem(self, problem: str) -> str:
        """
        Categorize a math problem by topic

        Args:
            problem: Math problem text

        Returns:
            Category name
        """
        try:
            response = self.categorize_chain.invoke({"problem": problem})
            category = response.get("text", "").strip().lower()
            # Ensure category is one of the recognized ones
            valid_categories = [
                "algebra", "calculus", "geometry", "probability", "statistics",
                "number_theory", "linear_algebra", "differential_equations", "trigonometry"
            ]
            if category not in valid_categories:
                category = "other"
            return category
        except Exception as e:
            print(f"Error categorizing problem: {e}")
            return "other"

    def record_feedback(self, query_id: int, feedback: Dict[str, Any]) -> None:
        """
        Record user feedback for a query

        Args:
            query_id: ID of the query
            feedback: Feedback data
        """
        if query_id in self.query_log:
            self.query_log[query_id]["feedback"] = feedback
