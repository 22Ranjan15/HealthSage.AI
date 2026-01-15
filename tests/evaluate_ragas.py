import os
import sys

# --- DISABLE LANGSMITH (Prevents 403 Errors) ---
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import pytest
import json
import asyncio
import numpy as np
import logging

from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas.run_config import RunConfig
from datasets import Dataset

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from backend.app.services.retrieval import RetrievalService
from config.config import Config
from utils.logging import LOG_FILE_PATH, get_logger
from utils.exceptions import CustomException

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
formatter = logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

logger = get_logger(__name__)

try:
    logger.info(f"🔌 Connecting to Hugging Face API: {Config.RAGAS_EVAL_MODEL}")

    base_llm = HuggingFaceEndpoint(
        repo_id=Config.RAGAS_EVAL_MODEL,
        max_new_tokens=512,
        top_k=10,
        temperature=0.1,
        repetition_penalty=1.03,
        huggingfacehub_api_token=Config.HUGGINGFACE_API_KEY,
        timeout=300 
    )
    evaluator_llm = ChatHuggingFace(llm=base_llm)

    logger.info(f"🧠 Loading Local Embeddings: {Config.EMBEDDING_MODEL_NAME}")
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL_NAME
    )
except Exception as e:
    logger.critical("Failed to connect to Hugging Face API. Check your HUGGINGFACE_API_KEY.")
    raise CustomException(e, sys)

@pytest.mark.asyncio
async def test_retrieval_quality():
    try:
        logger.info(">>> STARTING RAGAS EVALUATION <<<")

        if not os.path.exists(Config.GOLDEN_DATASET_PATH):
            error_msg = f"Golden dataset not found at {Config.GOLDEN_DATASET_PATH}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Loading dataset from: {Config.GOLDEN_DATASET_PATH}")
        with open(Config.GOLDEN_DATASET_PATH, "r") as f:
            data = json.load(f)

        logger.info("⚠️ Using first 3 items for test run.")
        data = data[:3]
        
        questions = [item["question"] for item in data]
        ground_truths = [item["ground_truth"] for item in data]

        service = RetrievalService()
        contexts = []
        
        logger.info(f"Running retrieval for {len(questions)} test cases...")
        
        for i, q in enumerate(questions):
            # Log progress every 5 queries
            if i % 5 == 0:
                logger.info(f"Processing query {i+1}/{len(questions)}: '{q}'")
            
            docs = await service.asearch(query=q, k=5)
            contexts.append([d.page_content for d in docs])

        # Preparation: Format for RAGAS
        dataset_dict = {
            "question": questions,
            "ground_truth": ground_truths,
            "contexts": contexts
        }
        dataset = Dataset.from_dict(dataset_dict)

        logger.info("Submitting results to RAGAS Judge (LLaMA)...")
        
        run_config = RunConfig(max_workers=1, timeout=600)
        results = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config
        )

        logger.info("--- RAGAS Evaluation Results ---")
        logger.info(results)
        
        # Handle different return types (Ragas v0.2 vs v0.1)
        raw_precision = results["context_precision"]

        if isinstance(raw_precision, list):
            precision_score = np.nanmean(raw_precision)
            logger.info(f"Calculated Average Precision from list: {precision_score}")
        else:
            precision_score = float(raw_precision)

        logger.info(f"Final Context Precision Score: {precision_score}")

        if precision_score < 0.7:
            failure_msg = f"❌ Retrieval Precision {precision_score:.2f} is below threshold 0.7"
            logger.error(failure_msg)
            pytest.fail(failure_msg)
        else:
            logger.info("✅ Retrieval Quality Passed Industry Standards.")

    except Exception as e:
        logger.critical("RAGAS Evaluation Script Crashed.")
        raise CustomException(e, sys)