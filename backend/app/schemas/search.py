from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- Request Model ---
class RetrievalRequest(BaseModel):
    query: str = Field(
        ..., 
        min_length=3, 
        max_length=500,
        description="The user's health-related question.",
        example="What are the early symptoms of diabetes?"
    )
    top_k: int = Field(
        default=10, 
        ge=1, 
        le=20, 
        description="Number of document chunks to retrieve."
    )

# --- Response Models ---
class DocumentChunk(BaseModel):
    content: str = Field(..., description="The main text content of the retrieved chunk.")
    source: str = Field(..., description="The filename of the source PDF.")
    score: Optional[float] = Field(None, description="Similarity score (0-1). Higher is better.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata like page numbers.")

class RetrievalResponse(BaseModel):
    results: List[DocumentChunk]
    total_found: int
    message: str = "Success"