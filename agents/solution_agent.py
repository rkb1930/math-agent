# agents/solution_agent.py
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import config


class MathSolutionAgent:
    """Agent for generating step-by-step math solutions"""

    def __init__(self):
        """Initialize the solution agent"""
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.2  # Lower temperature for more accurate math
        )

        # Prompt for solution generation from knowledge base
        self.kb_prompt_template = PromptTemplate(
            input_variables=["problem", "similar_problems", "similar_solutions"],
            template=""" 
            You are a brilliant mathematics professor who specializes in explaining complex concepts in simple terms. 
            You need to solve the following problem step by step:

            PROBLEM:
            {problem}

            I have found similar problems in our knowledge base that might help:

            SIMILAR PROBLEMS:
            {similar_problems}

            SIMILAR SOLUTIONS:
            {similar_solutions}

            Please provide a detailed step-by-step solution to the original problem. 
            Include mathematical reasoning at each step and explain the concepts clearly.
            Use LaTeX formatting for mathematical expressions where appropriate (surround with $ for inline math and $$ for display math).
            Ensure your solution is educational and helps the student understand the underlying concepts.
            """
        )

        # Prompt for solution generation from web search
        self.search_prompt_template = PromptTemplate(
            input_variables=["problem", "search_results"],
            template=""" 
            You are a brilliant mathematics professor who specializes in explaining complex concepts in simple terms.
            You need to solve the following problem step by step:

            PROBLEM:
            {problem}

            I have found the following information from web search that might help:

            SEARCH RESULTS:
            {search_results}

            Please provide a detailed step-by-step solution to the original problem.
            Include mathematical reasoning at each step and explain the concepts clearly.
            Use LaTeX formatting for mathematical expressions where appropriate (surround with $ for inline math and $$ for display math).
            Ensure your solution is educational and helps the student understand the underlying concepts.

            If the search results do not provide enough information to solve the problem confidently, 
            please state this clearly and provide your best approach to the problem with appropriate disclaimers.
            """
        )

        # Fallback prompt when neither KB nor search is helpful
        self.direct_prompt_template = PromptTemplate(
            input_variables=["problem"],
            template=""" 
            You are a brilliant mathematics professor who specializes in explaining complex concepts in simple terms.
            You need to solve the following problem step by step:

            PROBLEM:
            {problem}

            Please provide a detailed step-by-step solution to this problem.
            Include mathematical reasoning at each step and explain the concepts clearly.
            Use LaTeX formatting for mathematical expressions where appropriate (surround with $ for inline math and $$ for display math).
            Ensure your solution is educational and helps the student understand the underlying concepts.

            If you cannot solve the problem with high confidence, please state this clearly 
            and explain your approach as far as you can go.
            """
        )

        # Create chains
        self.kb_chain = LLMChain(llm=self.llm, prompt=self.kb_prompt_template)
        self.search_chain = LLMChain(llm=self.llm, prompt=self.search_prompt_template)
        self.direct_chain = LLMChain(llm=self.llm, prompt=self.direct_prompt_template)

    def generate_from_knowledge_base(self, problem: str, kb_results: List[Dict[str, Any]]) -> str:
        """
        Generate solution using knowledge base results

        Args:
            problem: Math problem to solve
            kb_results: Results from knowledge base

        Returns:
            Step-by-step solution
        """
        # Extract similar problems and solutions from knowledge base results
        similar_problems = "\n\n".join([
            f"Problem {i + 1}: {result['problem']}"
            for i, result in enumerate(kb_results)
        ])

        similar_solutions = "\n\n".join([
            f"Solution {i + 1}: {result['solution']}"
            for i, result in enumerate(kb_results)
        ])

        # Generate solution
        response = self.kb_chain.invoke({
            "problem": problem,
            "similar_problems": similar_problems,
            "similar_solutions": similar_solutions
        })

        return response.get("text", "")

    def generate_from_search(self, problem: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Generate solution using web search results

        Args:
            problem: Math problem to solve
            search_results: Results from web search

        Returns:
            Step-by-step solution
        """
        # Format search results for the prompt
        formatted_results = "\n\n".join([
            f"Result {i + 1} (Source: {result['title']}):\n{result['content']}"
            for i, result in enumerate(search_results)
        ])

        # Generate solution
        response = self.search_chain.invoke({
            "problem": problem,
            "search_results": formatted_results
        })

        return response.get("text", "")

    def generate_direct_solution(self, problem: str) -> str:
        """
        Generate solution directly when KB and search fail

        Args:
            problem: Math problem to solve

        Returns:
            Step-by-step solution
        """
        response = self.direct_chain.invoke({"problem": problem})
        return response.get("text", "")
