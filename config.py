import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Model Configuration
LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"

# Vector Database Settings
VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "localhost")
VECTOR_DB_PORT = int(os.getenv("VECTOR_DB_PORT", "6333"))
VECTOR_DB_COLLECTION = "math-knowledge"

# Application Settings
MAX_SEARCH_RESULTS = 3
SIMILARITY_THRESHOLD = 0.75
FEEDBACK_CATEGORIES = ["clarity", "correctness", "helpfulness"]
FEEDBACK_SCALE = 5
