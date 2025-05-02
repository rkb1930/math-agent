# Math Professor: Human-in-the-Loop Feedback Learning System

This is an Agentic-RAG architecture system that replicates a mathematical professor, providing step-by-step solutions to mathematical problems while learning from human feedback.

## Features

- **AI Gateway with Guardrails**: Ensures content is educational and math-focused
- **Knowledge Base**: Vector database of mathematical problems and solutions
- **Web Search**: Looks up solutions online when the knowledge base doesn't have relevant information
- **Human-in-the-Loop**: Collects feedback to improve solutions over time
- **DSPy Learning Module**: Uses feedback data to optimize solution generation
- **JEE Benchmark Evaluation**: Tests the system against standardized math problems

## Architecture

The system uses:
- **LangGraph** for agent workflow management
- **LlamaIndex** for knowledge retrieval and vector storage
- **Qdrant** as the vector database
- **Tavily API** for web search
- **DSPy** for human-feedback-based learning
- **Streamlit** for the user interface

## Setup Instructions

### Prerequisites

- Python 3.9+
- API keys for OpenAI and Tavily

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/math-professor.git
cd math-professor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the requirements:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

## Usage

1. Enter a math problem in the "Ask a Question" tab
2. The system will provide a step-by-step solution
3. Provide feedback to help the system improve
4. Check the sample problems for inspiration

## System Components

### Knowledge Base

The system uses a Mathematics Stack Exchange dataset stored in a Qdrant vector database. When a query is received, the system searches for similar problems in the knowledge base and retrieves relevant solutions.

Sample questions from the knowledge base:
1. "Solve the quadratic equation x² + 5x + 6 = 0"
2. "Find the derivative of f(x) = x³ - 4x² + 2x - 7"
3. "Calculate the area of a circle with radius 8 cm"

### Web Search

If the knowledge base doesn't contain relevant information, the system uses Tavily API to search the web for solutions. The search focuses on educational websites and mathematical resources.

Sample questions requiring web search:
1. "Find all values of x that satisfy 3sin(x) + 4cos(x) = 2 for 0 ≤ x ≤ 2π"
2. "Solve the differential equation dy/dx + 5y = e^(-5x)"
3. "Prove that for any positive integer n, the number n³ - n is divisible by 6"

### Human-in-the-Loop Feedback

The system implements a feedback mechanism where users can rate solutions on:
- Clarity
- Correctness
- Helpfulness

This feedback is used to improve the routing decisions and solution quality over time. The DSPy learning module analyzes high-quality solutions and adjusts the solution generation process accordingly.

## Evaluation

The system includes a JEE benchmark evaluator that tests the Math Agent against a subset of JEE problems. Results are stored in the `evaluation_results` directory.

## Project Structure

```
math_agent/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Dependencies
├── config.py               # Configuration settings
├── agents/
│   ├── __init__.py
│   ├── router.py           # Query routing logic
│   ├── knowledge_agent.py  # Knowledge base retrieval
│   ├── search_agent.py     # Web search functionality
│   └── solution_agent.py   # Solution generation
├── feedback/
│   ├── __init__.py
│   ├── collector.py        # Human feedback collection
│   └── learner.py          # DSPy-based learning module
├── data/
│   ├── __init__.py
│   ├── loader.py           # Data loading utilities
│   └── vectordb.py         # Vector database operations
└── utils/
    ├── __init__.py
    ├── guardrails.py       # Input/output guardrails
    └── evaluation.py       # JEE benchmark evaluation
```

## Future Improvements

1. Add support for image processing to solve handwritten problems
2. Implement comparison learning with multiple solution approaches
3. Add specific modules for different math domains (calculus, algebra, etc.)
4. Enhance the feedback system with targeted improvement suggestions
5. Integrate with popular learning management systems

## License

MIT License

## Contact

For questions or support, please contact [iamrohankumarbehera@gmail.com]
