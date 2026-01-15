import sys
import os
from typing import List

# --- LangChain & Qdrant ---
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings 
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from flashrank import Ranker, RerankRequest

# --- Custom Modules ---
from utils.sparse import FastEmbedSparse
from config.config import Config
from utils.logging import get_logger
from utils.exceptions import CustomException

logger = get_logger(__name__)

# --- Global Cache (Singleton) ---
_dense_model = None
_sparse_model = None
_reranker_model = None

def get_models():
    global _dense_model, _sparse_model, _reranker_model

    # 1. Dense (Semantic)
    if _dense_model is None:
        logger.info(f"Loading Dense: {Config.EMBEDDING_MODEL_NAME}")
        _dense_model = FastEmbedEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
    
    # 2. Sparse (Keywords)
    if _sparse_model is None:
        logger.info(f"Loading Sparse: {Config.SPARSE_MODEL_NAME}")
        _sparse_model = FastEmbedSparse(model_name=Config.SPARSE_MODEL_NAME)

    # 3. Reranker (Precision)
    if _reranker_model is None:
        logger.info("Loading Reranker (FlashRank)")
        cache_path = os.environ.get("FASTEMBED_CACHE_PATH", "/tmp/fastembed_cache")
        _reranker_model = Ranker(model_name=Config.RERANKER_MODEL_NAME, cache_dir=cache_path)
        
    return _dense_model, _sparse_model, _reranker_model


class RetrievalService:
    def __init__(self):
        try:
            logger.info("Initializing Retrieval Service Connection...")
            
            # --- ADD THIS DEBUG BLOCK ---
            print(f"DEBUG: QDRANT_URL from Config is: '{Config.QDRANT_URL}'")
            print(f"DEBUG: API Key length is: {len(Config.QDRANT_URL) if Config.QDRANT_URL else 0}")
            # ----------------------------
            
            self.client = QdrantClient(
                url=Config.QDRANT_URL, 
                api_key=Config.QDRANT_API_KEY
            )
            
            # Load Models
            dense, sparse, reranker = get_models()
            self.reranker = reranker

            # Setup Hybrid Store
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=Config.COLLECTION_NAME,
                embedding=dense,
                sparse_embedding=sparse,
                vector_name="dense",
                sparse_vector_name="sparse",
                retrieval_mode=RetrievalMode.HYBRID 
            )
        except Exception as e:
            logger.error("Failed to initialize Retrieval Service.")
            raise CustomException(e, sys)

    async def asearch(self, query: str, k: int = 5) -> List[Document]:
        """
        Agentic Pipeline:
        1. Hybrid Broad Search (Vectors + Keywords) -> Fetch 25
        2. FlashRank Reranking (Context Awareness) -> Filter to k
        """
        try:
            logger.info(f"Pipeline Start: '{query}'")

            # --- Step 1: Hybrid Search ---
            logger.info(f"Executing Hybrid Search for top {k * 5} candidates...")
            # FETCH 1: Broad Net (Recall)
            fetch_count = max(k * 5, 25) 
            candidates = await self.vector_store.asimilarity_search(
                query=query, 
                k=fetch_count
            )

            if not candidates:
                logger.warning("No candidates retrieved from Hybrid Search.")
                return []
            logger.info(f"Hybrid Search fetched {len(candidates)} candidates.")

            # --- Step 2: Reranking ---
            # FETCH 2: Reranking (Precision)
            logger.info(f"Reranking {len(candidates)} candidates...")
            
            # Convert to FlashRank Input
            passages = [
                {"id": str(i), "text": doc.page_content, "meta": doc.metadata} 
                for i, doc in enumerate(candidates)
            ]

            rerank_request = RerankRequest(query=query, passages=passages)
            ranked_results = self.reranker.rerank(rerank_request)

            # --- Step 3: Reconstruction ---
            final_docs = []
            for res in ranked_results[:k]: # Take only top K
                doc = Document(
                    page_content=res["text"],
                    metadata=res["meta"]
                )
                doc.metadata["score"] = res["score"] # Add confidence score
                final_docs.append(doc)
            
            return final_docs            
        except Exception as e:
            raise CustomException(e, sys)