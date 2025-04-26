import os
from typing import List, Dict, Any

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

import config


class MathKnowledgeBase:
    """Vector database for math knowledge base using Qdrant and LlamaIndex"""

    def __init__(self):
        """Initialize the knowledge base"""
        self.client = QdrantClient(
            url=config.VECTOR_DB_HOST,
            port=config.VECTOR_DB_PORT
        )

        self.embed_model = OpenAIEmbedding(
            model=config.EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY
        )

        # Ensure collection exists
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if config.VECTOR_DB_COLLECTION not in collection_names:
            self._create_collection()

    def _create_collection(self):
        """Create a new collection in Qdrant"""
        self.client.create_collection(
            collection_name=config.VECTOR_DB_COLLECTION,
            vectors_config=models.VectorParams(
                size=1536,  # Dimension for OpenAI embeddings
                distance=models.Distance.COSINE
            )
        )

    def load_math_dataset(self, data_path: str):
        """
        Load math dataset into the vector database

        Args:
            data_path: Path to the math dataset CSV
        """
        df = pd.read_csv(data_path)

        documents = []
        for idx, row in df.iterrows():
            try:
                problem = row.get('problem', '')
                solution = row.get('solution', '')
                metadata = {
                    'source': 'math_exchange',
                    'difficulty': row.get('difficulty', 'medium'),
                    'topic': row.get('topic', 'general'),
                    'has_solution': bool(solution)
                }

                doc = Document(
                    text=f"Problem: {problem}\nSolution: {solution}",
                    metadata=metadata
                )
                documents.append(doc)

                if idx % 100 == 0:
                    print(f"Processed {idx} documents")

            except Exception as e:
                print(f"Error processing document {idx}: {e}")

        # Store into Qdrant
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=config.VECTOR_DB_COLLECTION
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self.embed_model
        )

        print(f"Loaded {len(documents)} math problems into the knowledge base")
        return index

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar math problems in the knowledge base

        Args:
            query: The math problem to search for
            top_k: Number of results to return

        Returns:
            List of relevant math problems and solutions
        """
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=config.VECTOR_DB_COLLECTION
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=self.embed_model
        )

        retriever = index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query)

        formatted_results = []
        for result in results:
            text_parts = result.text.split("\nSolution: ")
            problem = text_parts[0].replace("Problem: ", "")
            solution = text_parts[1] if len(text_parts) > 1 else ""

            formatted_results.append({
                "problem": problem.strip(),
                "solution": solution.strip(),
                "score": result.score,
                "metadata": result.metadata
            })

        return formatted_results


def get_sample_math_dataset():
    """Download a sample math dataset if it doesn't exist"""
    from datasets import load_dataset

    output_path = "data/math_dataset.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        return output_path

    dataset = load_dataset("csv", data_files={
        "train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.csv"
    })
    df = dataset["train"].to_pandas()

    df = df[["question", "answer"]]
    df.columns = ["problem", "solution"]
    df.to_csv(output_path, index=False)

    return output_path
