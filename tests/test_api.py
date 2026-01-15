from fastapi.testclient import TestClient
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from backend.main import app
from config.config import Config
from utils.logging import get_logger, LOG_FILE_PATH 
from utils.exceptions import CustomException

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE_PATH)
formatter = logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

logger = get_logger(__name__)
client = TestClient(app)

def test_health_check_endpoint():
    try:
        logger.info(">>> TEST START: Health Check (GET /)")
        
        response = client.get("/")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        json_data = response.json()
        assert json_data["status"] == "active"
        assert json_data["service"] == "HealthSage.AI Backend"
        
        logger.info("✅ Health Check Passed")
    except Exception as e:
        logger.error("Health Check Failed")
        raise CustomException(e, sys)

def test_search_endpoint_integration():
    try:
        logger.info(">>> TEST START: Search Integration (POST /api/v1/search)")
        
        payload = {
            "query": "What are the symptoms of Asthma?",
            "top_k": 3
        }
        
        response = client.post("/api/v1/search", json=payload)
        
        logger.info(f"API Response Code: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"API Error Response: {response.text}")
        assert response.status_code == 200 

        data = response.json()
        assert "results" in data, "Response missing 'results' key"
        assert "total_found" in data, "Response missing 'total_found' key"
        assert isinstance(data["results"], list), "'results' must be a list"

        if len(data["results"]) > 0:
            first_doc = data["results"][0]
            score = first_doc.get("score")
            
            assert isinstance(score, (float, type(None))), f"Invalid score type: {type(score)}"
            logger.info(f"Score type validation passed: {score} is {type(score)}")
        
        logger.info("✅ Search Endpoint Integration Test Passed")

    except Exception as e:
        logger.critical("Search Endpoint Test Failed")
        raise CustomException(e, sys)

def test_search_validation_error():
    try:
        logger.info(">>> TEST START: Search Validation (Invalid Payload)")

        invalid_payload = {
            "top_k": 5
        }
        response = client.post("/api/v1/search", json=invalid_payload)
        
        logger.info(f"Response Code: {response.status_code}")
        
        assert response.status_code == 422, "API should reject missing query with 422"
        
        logger.info("✅ Validation Logic Passed")

    except Exception as e:
        logger.error("Validation Test Failed")
        raise CustomException(e, sys)