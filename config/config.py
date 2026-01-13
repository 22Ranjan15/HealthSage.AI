import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # --- Project Root Calculation ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # --- Centralized Paths ---
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

    # --- Qdrant (Vector DB) ---
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = "healthsage_medical_docs"

    # --- Gemini API ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # --- Models ---
    # Dense: Understanding meaning (Semantic)
    EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
    VECTOR_SIZE = 768

    # Sparse: Understanding exact keywords (SPLADE)
    SPARSE_MODEL_NAME = "prithivida/Splade_pp_en_v1"

    # Reranker: Filtering results (FlashRank)
    RERANKER_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"
    
    # Generative: Answer generation (Gemini)
    LLM_MODEL_NAME = "gemini-3-flash-preview"

    # --- Chunking Strategies ---
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200


    @staticmethod
    def validate():
        """Optional: Check if critical environment variables are set."""
        if not Config.QDRANT_URL or not Config.QDRANT_API_KEY:
            raise ValueError("CRITICAL: QDRANT_URL or QDRANT_API_KEY missing from .env file.")

Config.validate()