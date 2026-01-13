import os
from fastembed import TextEmbedding, SparseTextEmbedding
from flashrank import Ranker

# Configuration
DENSE_MODEL = "BAAI/bge-base-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_pp_en_v1"
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"


def download_all():
    print(">>> STARTING MODEL BAKING...")
    
    # 1. Determine Cache Path
    cache_dir = os.environ.get("FASTEMBED_CACHE_PATH", "/tmp/fastembed_cache")
    print(f"Target Directory: {cache_dir}")

    # 2. Download Dense Model (Vectors)
    print(f"Downloading Dense: {DENSE_MODEL}...")
    TextEmbedding(model_name=DENSE_MODEL, cache_dir=cache_dir)

    # 3. Download Sparse Model (Keywords) - CRITICAL FIX
    print(f"Downloading Sparse: {SPARSE_MODEL}...")
    SparseTextEmbedding(model_name=SPARSE_MODEL, cache_dir=cache_dir)

    # 4. Download Reranker
    print(f"Downloading Reranker: {RERANKER_MODEL}...")
    Ranker(model_name=RERANKER_MODEL, cache_dir=cache_dir)
    
    print(">>> ALL MODELS BAKED SUCCESSFULLY.")

if __name__ == "__main__":
    download_all()