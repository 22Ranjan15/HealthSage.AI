import os
import sys
from typing import List
import time

# --- LangChain & Qdrant Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings 
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

# --- Custom Modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from config.config import Config
from utils.logging import get_logger
from utils.exceptions import CustomException
from utils.sparse import FastEmbedSparse

logger = get_logger(__name__)


class ETLPipeline:
    def __init__(self):
        try:
            logger.info("Initializing ETL Pipeline components...")
            self.client = QdrantClient(
                url=Config.QDRANT_URL, 
                api_key=Config.QDRANT_API_KEY,
                timeout=120.0 
            )

            # 1. Load Dense Model (Standard)
            logger.info(f"Loading Dense Embedding model: {Config.EMBEDDING_MODEL_NAME}")
            self.dense_embeddings = FastEmbedEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
            
            # 2. Load Sparse Model (Custom Wrapper)
            logger.info(f"Loading Sparse Embedding model: {Config.SPARSE_MODEL_NAME}")
            self.sparse_embeddings = FastEmbedSparse(model_name=Config.SPARSE_MODEL_NAME)

            logger.info("Embedding models loaded successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def extract(self) -> List[Document]:
        try:
            logger.info(f"Step 1: EXTRACT - Loading files from {Config.DATA_PATH}")
            documents = []
            
            if not os.path.exists(Config.DATA_PATH):
                raise FileNotFoundError(f"Directory {Config.DATA_PATH} not found.")

            files = [f for f in os.listdir(Config.DATA_PATH) if f.endswith(".pdf")]
            if not files:
                logger.warning(f"No PDF files found in {Config.DATA_PATH}. Please add data.")
                return []

            for filename in files:
                file_path = os.path.join(Config.DATA_PATH, filename)
                logger.info(f"Processing file: {filename}")
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = filename
                    
                    documents.extend(docs)
                except Exception as file_error:
                    logger.error(f"Failed to load {filename}: {file_error}")
                    continue
            
            logger.info(f"Extracted {len(documents)} total pages.")
            return documents
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, documents: List[Document]) -> List[Document]:
        try:
            logger.info("Step 2: TRANSFORM - Chunking documents")
            if not documents:
                return []

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            chunks = text_splitter.split_documents(documents)

            logger.info(f"Transformed source pages into {len(chunks)} processed chunks.")
            return chunks
        except Exception as e:
            raise CustomException(e, sys)

    def load(self, chunks: List[Document]):
        try:
            logger.info(f"Step 3: LOAD - Uploading to Qdrant Configured Collection: {Config.COLLECTION_NAME}")
            if not chunks:
                logger.warning("No chunks to load. Pipeline skipping database update.")
                return

            # 1. Re-Create Collection
            if self.client.collection_exists(Config.COLLECTION_NAME):
                logger.warning("Deleting old collection to apply Hybrid Schema...")
                self.client.delete_collection(Config.COLLECTION_NAME)

            self.client.create_collection(
                collection_name=Config.COLLECTION_NAME,
                vectors_config={
                    "dense": models.VectorParams(size=Config.VECTOR_SIZE, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )
            logger.info("Collection created.")

            # 2. Initialize Store
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=Config.COLLECTION_NAME,
                embedding=self.dense_embeddings,
                sparse_embedding=self.sparse_embeddings,
                vector_name="dense",
                sparse_vector_name="sparse",
                retrieval_mode=RetrievalMode.HYBRID
            )

            # 3. Batch Upload Logic -- Prevent Overloads
            batch_size = 20 
            total_chunks = len(chunks)
            logger.info(f"Starting Upload: {total_chunks} chunks in batches of {batch_size}...")

            for i in range(0, total_chunks, batch_size):
                batch = chunks[i : i + batch_size]
                if not batch: 
                    continue

                # --- RETRY LOGIC START ---
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        vector_store.add_documents(batch)
                        logger.info(f"Progress: Uploaded {min(i + batch_size, total_chunks)} / {total_chunks} chunks.")
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = 5 * (attempt + 1) # Wait 5s, then 10s
                            logger.warning(f"Network blip on batch {i}. Retrying in {wait_time}s... Error: {e}")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Permanent failure uploading batch starting at index {i}")
                            raise e
                # --- RETRY LOGIC END ---

            logger.info(">>> PIPELINE COMPLETE: All data successfully indexed.")
        except Exception as e:
            raise CustomException(e, sys)

    def run(self):
        try:
            logger.info(">>> ETL PIPELINE STARTED <<<")
            
            # 1. EXTRACT
            raw_docs = self.extract()
            
            # 2. TRANSFORM
            chunks = self.transform(raw_docs)
            
            # 3. LOAD
            self.load(chunks)

        except CustomException as ce:
            logger.error(f"Pipeline stopped due to custom logic error: {ce}")
        except Exception as e:
            logger.critical(f"Pipeline crashed entirely: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    etl = ETLPipeline()
    etl.run()