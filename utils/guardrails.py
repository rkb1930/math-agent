import re
from typing import Dict, Any, Tuple, List


class MathGuardrails:
    """Input and output guardrails for the Math Agent"""

    # Keywords that indicate mathematical content
    MATH_KEYWORDS = [
        "solve", "calculate", "equation", "formula", "integral", "derivative",
        "algebra", "calculus", "geometry", "probability", "statistics",
        "matrix", "vector", "function", "theorem", "proof", "trigonometry",
        "polynomial", "logarithm", "exponent", "+", "-", "*", "/", "^", "=",
        "sin", "cos", "tan", "log", "ln", "limit", "differentiate", "integrate",
        "factor", "expand", "simplify", "solve for", "determine", "graph"
    ]

    # Content to filter out (non-educational)
    DISALLOWED_CONTENT = [
        "cheat", "exam solutions", "test answers", "do my homework",
        "solve this for me", "complete my assignment", "plagiarism", "unethical"
    ]

    @staticmethod
    def validate_input(query: str) -> Tuple[bool, str]:
        """
        Validates if the input query is educational and math-related

        Args:
            query: User input query

        Returns:
            Tuple of (is_valid, message)
        """
        # Check for disallowed content
        if any(phrase in query.lower() for phrase in MathGuardrails.DISALLOWED_CONTENT):
            return False, "I can help you learn math, but I won't do your homework or exams for you."

        # Check if query is math-related
        if not any(keyword in query.lower() for keyword in MathGuardrails.MATH_KEYWORDS):
            if len(query.split()) > 3:  # If it's a substantial query but not obviously math
                # Let it pass but flag it for human review
                return True, "Note: This query doesn't appear to be explicitly math-related."
            else:
                # Short query with no math keywords
                return False, "Please ask a math-related question. I'm here to help with mathematics."

        return True, ""

    @staticmethod
    def sanitize_output(response: str) -> str:
        """
        Ensures the output is appropriate and educational

        Args:
            response: Generated response

        Returns:
            Sanitized response
        """
        # Ensure mathematical notation is preserved (especially LaTeX)
        # Add proper context for educational purposes
        if not response.startswith("Here's a step-by-step solution") and \
                not response.startswith("Let me solve this") and \
                not response.startswith("Let's work through this"):
            response = "Here's how to approach this math problem:\n\n" + response

        # Add learning context if not present
        if "understand why" not in response.lower() and "concept behind" not in response.lower():
            response += "\n\nUnderstanding the concepts behind this solution is important. If you have questions about any step, feel free to ask!"

        # Add a reminder about the importance of practice
        response += "\n\nRemember, practice is key to mastering math. Keep working on similar problems to improve!"

        return response
