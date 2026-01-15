import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # --- Project Root Calculation ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

    # --- Centralized Paths ---
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
    GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

    # --- Qdrant (Vector DB) ---
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = "healthsage_medical_docs"

    # --- Gemini API ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # --- HuggingFace API ---
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
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
    RATE_LIMIT_RPM = 15
    RATE_LIMIT_INTERVAL = 60 / RATE_LIMIT_RPM

    # RAGAS Evaluation Model
    RAGAS_EVAL_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

    # --- Chunking Strategies ---
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200


    @staticmethod
    def validate():
        if not Config.QDRANT_URL or not Config.QDRANT_API_KEY:
            raise ValueError("CRITICAL: QDRANT_URL or QDRANT_API_KEY missing.")
        if not Config.HUGGINGFACE_API_KEY:
            raise ValueError("CRITICAL: HUGGINGFACE_API_KEY is missing. Ragas cannot run.")

Config.validate()