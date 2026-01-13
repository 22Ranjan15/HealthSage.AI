import os
import sys
from typing import List

# --- LangChain & Qdrant Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

# --- Custom Modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from config.config import Config
from utils.logging import get_logger
from utils.exceptions import CustomException

# Initialize Logger
logger = get_logger(__name__)


class ETLPipeline:
    """
    Extract, Transform, Load (ETL) Pipeline for HealthSage AI.
    - EXTRACT: Reads medical PDFs from data/raw.
    - TRANSFORM: Chunks text using recursive character splitting.
    - LOAD: Embeds chunks via BGE-Base and upserts to Qdrant Cloud.
    """

    def __init__(self):
        try:
            logger.info("Initializing ETL Pipeline components...")
            
            # 1. Initialize Qdrant Client
            self.client = QdrantClient(
                url=Config.QDRANT_URL, 
                api_key=Config.QDRANT_API_KEY
            )
            
            # 2. Initialize Usage of Embedding Model
            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL_NAME}")
            self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def extract(self) -> List[Document]:
        """
        Loads PDF files from the configured data directory.
        """
        try:
            logger.info(f"Step 1: EXTRACT - Loading files from {Config.DATA_PATH}")
            documents = []
            
            if not os.path.exists(Config.DATA_PATH):
                raise FileNotFoundError(f"Directory {Config.DATA_PATH} not found.")

            # List PDF files
            files = [f for f in os.listdir(Config.DATA_PATH) if f.endswith(".pdf")]
            
            if not files:
                logger.warning(f"No PDF files found in {Config.DATA_PATH}. Please add data.")
                return []

            # Load each file
            for filename in files:
                file_path = os.path.join(Config.DATA_PATH, filename)
                logger.info(f"Processing file: {filename}")
                
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    
                    # Add source metadata to every page for citation in the UI later
                    for doc in docs:
                        doc.metadata["source"] = filename
                    
                    documents.extend(docs)
                except Exception as file_error:
                    logger.error(f"Failed to load {filename}: {file_error}")
                    # We continue to the next file instead of crashing the whole pipeline
                    continue
            
            logger.info(f"Extracted {len(documents)} total pages.")
            return documents

        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents into vector-ready chunks.
        """
        try:
            logger.info("Step 2: TRANSFORM - Chunking documents")
            
            if not documents:
                return []

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
                separators=Config.SEPARATORS
            )
            
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Transformed source pages into {len(chunks)} processed chunks.")
            return chunks

        except Exception as e:
            raise CustomException(e, sys)

    def load(self, chunks: List[Document]):
        """
        Generates embeddings and upserts them to Qdrant Cloud.
        """
        try:
            logger.info(f"Step 3: LOAD - Uploading to Qdrant Configured Collection: {Config.COLLECTION_NAME}")
            
            if not chunks:
                logger.warning("No chunks to load. Pipeline skipping database update.")
                return

            # Check if Collection Exists
            if not self.client.collection_exists(Config.COLLECTION_NAME):
                logger.info(f"Collection '{Config.COLLECTION_NAME}' does not exist. Creating now...")
                self.client.create_collection(
                    collection_name=Config.COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=Config.VECTOR_SIZE,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info("Collection created.")

            # Batch Upsert using LangChain's optimized wrapper
            # prefer_grpc=True is faster for cloud connections
            QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                url=Config.QDRANT_URL,
                api_key=Config.QDRANT_API_KEY,
                collection_name=Config.COLLECTION_NAME,
                prefer_grpc=True
            )
            
            logger.info("Successfully loaded all chunks into the Vector Database.")

        except Exception as e:
            raise CustomException(e, sys)

    def run(self):
        """
        Orchestrator to run the full pipeline.
        """
        try:
            logger.info(">>> ETL PIPELINE STARTED <<<")
            
            # 1. EXTRACT
            raw_docs = self.extract()
            
            # 2. TRANSFORM
            chunks = self.transform(raw_docs)
            
            # 3. LOAD
            self.load(chunks)
            
            logger.info(">>> ETL PIPELINE COMPLETED SUCCESSFULLY <<<")
            
        except CustomException as ce:
            logger.error(f"Pipeline stopped due to custom logic error: {ce}")
        except Exception as e:
            logger.critical(f"Pipeline crashed entirely: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    etl = ETLPipeline()
    etl.run()