from typing import List
from langchain_core.embeddings import Embeddings
from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

class FastEmbedSparse(Embeddings):
    def __init__(self, model_name: str):
        self.model = SparseTextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        results = list(self.model.embed(texts))
        
        return [
            SparseVector(
                indices=list(res.indices), 
                values=list(res.values)
            ) 
            for res in results
        ]

    def embed_query(self, text: str) -> SparseVector:
        res = list(self.model.embed([text]))[0]
        return SparseVector(
            indices=list(res.indices), 
            values=list(res.values)
        )