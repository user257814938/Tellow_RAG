import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration for the RAG system."""
    
    # ==========================================================================
    # 1. API KEYS & ENVIRONMENT
    # ==========================================================================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing in .env file")

    # ==========================================================================
    # 2. DOCUMENT PROCESSING (Chunking)
    # ==========================================================================
    # Size of each text chunk (in characters)
    CHUNK_SIZE = 1000
    # Overlap between consecutive chunks to maintain context
    CHUNK_OVERLAP = 200

    # ==========================================================================
    # 3. AI MODELS
    # ==========================================================================
    # Model used for generating vector embeddings
    EMBEDDING_MODEL = "text-embedding-3-small"
    # Model used for generating the final answer
    LLM_MODEL = "gpt-4o"
    
    # ==========================================================================
    # 4. VECTOR DATABASE
    # ==========================================================================
    LANCEDB_URI = "data/lancedb"
    TABLE_NAME = "docling_docs"
    
    # ==========================================================================
    # 5. RETRIEVAL PARAMETERS
    # ==========================================================================
    # Number of documents/chunks to retrieve per query
    SEARCH_K = 4
