"""Configuration constants for the RAG application."""

import os

# Disable tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Environment-based configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")  # Placeholder for local llama-server
OPENAI_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8080")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "dummy")
CHROMA_PATH = os.getenv("CHROMA_PATH", "vectorstore")
PDF_PATH = os.getenv("PDF_PATH", "pdf")

# Input validation
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "10000"))
ALLOWED_SEARCH_TYPES = {"mmr", "similarity", "hybrid"}

# RAG configuration
RETRIEVER_K = 5  # Number of final documents to return
RETRIEVER_FETCH_K = 15  # Number of candidates to fetch for MMR
MMR_LAMBDA = 0.7  # Balance between relevance (1.0) and diversity (0.0)
CHAT_HISTORY_LIMIT = 5  # Number of recent messages to include in context

# Advanced RAG configuration
HYBRID_ALPHA_DEFAULT = 0.7  # 70% semantic, 30% keyword
HYBRID_ALPHA_UI_DEFAULT = int(HYBRID_ALPHA_DEFAULT * 100)  # For UI slider (0-100)
HYBRID_INITIAL_K = 20  # Retrieve more candidates for hybrid
RERANK_INITIAL_K = 20  # Retrieve more candidates before re-ranking
RERANK_TOP_K = RETRIEVER_K  # Final number after re-ranking

# Embedding model configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DEVICE = "cpu"

# Text splitter configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# Reranker model
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# RAG prompt template
RAG_PROMPT_TEMPLATE = """Answer the question naturally and conversationally based on the provided context. Be direct and informative - if the answer is in the context, state it clearly without unnecessary formal structure or sections. Write as if you're explaining to a colleague.

Context:
{context}

Question: {question}

Previous conversation:
{chat_history}

Answer:"""



