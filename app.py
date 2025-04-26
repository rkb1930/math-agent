import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from transformers import pipeline

# Initialize Hugging Face model pipeline
@st.cache_resource
def get_huggingface_model():
    # Assuming we're using a general math-solving model, adjust as needed.
    return pipeline('text2text-generation', model="google/flan-t5-large")

# Set up page
st.set_page_config(
    page_title="Math Professor - Human-in-the-Loop Learning",
    page_icon="➗",
    layout="wide"
)

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "current_query_id" not in st.session_state:
    st.session_state.current_query_id = None
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# Title and explanation
st.title("🧮 Math Professor AI")
st.markdown("""
This AI system acts like a math professor, providing step-by-step solutions to your math problems.
It learns from your feedback to improve over time!

**Features:**
- Knowledge base of mathematical problems and solutions
- Web search for problems not in the knowledge base
- Human-in-the-loop feedback learning
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Math Professor** uses an Agentic-RAG architecture to:
    1. Check a knowledge base for similar problems
    2. Search the web if needed
    3. Generate detailed step-by-step solutions
    4. Learn from your feedback
    """)

    st.divider()

    # Admin functions
    st.header("Admin Functions")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Retrain Model"):
            # Retraining logic, if needed (depending on how you want to implement this)
            st.warning("Retraining not implemented with Hugging Face models in this version.")

    with col2:
        if st.button("Run Evaluation"):
            # Run evaluation logic for Hugging Face model (you may implement specific evaluation)
            st.warning("Evaluation not implemented with Hugging Face models in this version.")

    # Feedback stats
    st.divider()
    st.header("Feedback Statistics")
    # Placeholder for feedback statistics
    st.metric("Total Feedback Collected", 0)

# Main area
tab1, tab2, tab3 = st.tabs(["Ask a Question", "Sample Problems", "History"])

# Tab 1: Ask a Question
with tab1:
    st.header("Ask Any Math Question")

    # Input form
    with st.form("math_form"):
        query = st.text_area("Enter your math problem:", height=100,
                             placeholder="Example: Solve the quadratic equation x² + 5x + 6 = 0")
        submitted = st.form_submit_button("Solve")

    # Process query
    if submitted and query:
        # Get solution from Hugging Face model
        model = get_huggingface_model()

        with st.spinner("Solving the problem..."):
            response = model(query)

            # Process response
            solution = response[0]['generated_text']
            metadata = {"path_taken": "model_knowledge", "confidence": 1.0}

            # Store in session state
            query_id = hash(query)
            st.session_state.current_query_id = query_id
            st.session_state.query_history.append({
                "id": query_id,
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "solution": solution,
                "metadata": metadata
            })
            st.session_state.show_feedback = True
            st.session_state.feedback_submitted = False

        # Display solution
        st.subheader("Solution")
        st.markdown(solution)

        # Show source information
        st.divider()
        st.caption("Solution Method")

        method_map = {
            "model_knowledge": "🧠 Model Knowledge"
        }

        method = metadata.get("path_taken", "model_knowledge")
        st.info(f"{method_map.get(method, method)} (Confidence: {metadata.get('confidence', 0):.2f})")

        # Feedback form
        if st.session_state.show_feedback and not st.session_state.feedback_submitted:
            st.divider()
            st.subheader("Provide Feedback")

            with st.form("feedback_form"):
                st.write("Rate the solution:")
                col1, col2, col3 = st.columns(3)

                with col1:
                    clarity = st.slider("Clarity", 1, 5, 3)

                with col2:
                    correctness = st.slider("Correctness", 1, 5, 3)

                with col3:
                    helpfulness = st.slider("Helpfulness", 1, 5, 3)

                comments = st.text_area("Comments (optional):",
                                        placeholder="Any additional feedback on the solution...")
                submit_feedback = st.form_submit_button("Submit Feedback")

            if submit_feedback:
                # Collect feedback (Modify as necessary for your system)
                st.success("Thank you for your feedback! It helps improve the system.")
                st.session_state.feedback_submitted = True

# Tab 2: Sample Problems
with tab2:
    st.header("Sample Problems to Try")

    sample_problems = [
        {"title": "Quadratic Equation", "problem": "Solve the quadratic equation: 2x² - 7x + 3 = 0", "difficulty": "Easy", "category": "Algebra"},
        {"title": "Derivative", "problem": "Find the derivative of f(x) = x³ - 4x² + 2x - 7", "difficulty": "Medium", "category": "Calculus"},
        {"title": "Probability", "problem": "A bag contains 4 red marbles and 5 blue marbles. If two marbles are drawn without replacement, what is the probability that both are red?", "difficulty": "Medium", "category": "Probability"},
        {"title": "Geometry", "problem": "Find the area of a circle with radius 6 cm.", "difficulty": "Easy", "category": "Geometry"},
        {"title": "Linear Algebra", "problem": "Find the eigenvalues of the matrix [[3, 1], [2, 2]]", "difficulty": "Hard", "category": "Linear Algebra"}
    ]

    # Display as cards
    cols = st.columns(3)
    for i, problem in enumerate(sample_problems):
        with cols[i % 3]:
            st.markdown(f"### {problem['title']}")
            st.markdown(f"**{problem['category']}** • {problem['difficulty']}")
            st.markdown(f"{problem['problem']}")
            if st.button("Try This", key=f"sample_{i}"):
                # Set as current query
                st.session_state.sample_query = problem['problem']
                st.rerun()

# Tab 3: History
with tab3:
    st.header("Your Question History")

    if not st.session_state.query_history:
        st.info("Your question history will appear here.")
    else:
        for i, item in enumerate(reversed(st.session_state.query_history)):
            with st.expander(
                    f"{item['query'][:50]}... ({datetime.fromisoformat(item['timestamp']).strftime('%H:%M:%S')})"):
                st.markdown(f"**Query:** {item['query']}")
                st.markdown(f"**Solution:**\n{item['solution']}")

                method = item['metadata'].get('path_taken', 'model_knowledge')
                st.caption(f"Method: {method_map.get(method, method)}")

# Footer
st.divider()
st.caption("Math Professor AI • Human-in-the-Loop Learning System • ")
