from fastapi import APIRouter, HTTPException, Depends, status
from typing import Any

from backend.app.schemas.search import RetrievalRequest, RetrievalResponse, DocumentChunk

from backend.app.services.retrieval import RetrievalService
from utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# --- Dependency Injection ---
def get_service():
    return RetrievalService()

@router.post(
    "/search", 
    response_model=RetrievalResponse, 
    status_code=status.HTTP_200_OK,
    summary="Retrieve Relevant Medical Documents",
    description="Performs semantic search on the Qdrant Vector Database to find relevant context."
)
async def search_documents(
    payload: RetrievalRequest,
    service: RetrievalService = Depends(get_service)
) -> Any:
    """
    Core Retrieval Endpoint (Phase 3 Spec).
    """
    try:
        logger.info(f"API Request: query='{payload.query}' | top_k={payload.top_k}")
        
        # 1. Execute Search (Async for high concurrency)
        raw_documents = await service.asearch(
            query=payload.query, 
            k=payload.top_k
        )
        
        # 2. Transform Logic (Domain -> API)
        api_results = []
        for doc in raw_documents:
            # Extract score if available (Standard Search), maps to None if using MMR
            score = doc.metadata.get("score", None)
            
            api_results.append(
                DocumentChunk(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "Unknown Source"),
                    score=score,
                    metadata=doc.metadata
                )
            )

        logger.info(f"Successfully returned {len(api_results)} documents.")
        
        return RetrievalResponse(
            results=api_results,
            total_found=len(api_results),
            message="Retrieval successful"
        )

    except Exception as e:
        logger.error(f"Critical Endpoint Error: {str(e)}")
        # Hide detailed error trace from user, but keep it in logs
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An internal error occurred while processing the search request."
        )