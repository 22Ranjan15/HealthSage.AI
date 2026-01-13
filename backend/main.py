from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

# Import Config to read environment
from config.config import Config

# Import Routers
from backend.app.api.endpoints import router as api_router

# 1. Initialize Application
app = FastAPI(
    title="HealthSage AI API",
    description="Serverless RAG API for Medical Question Answering",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 2. Configure CORS (Cross-Origin Resource Sharing)
# Required for Phase 6 (React Frontend) to talk to Phase 3 (API)
app.add_middleware(
    CORSMiddleware,
    # In production/deployment, replace ["*"] with specific domains (e.g., your-vercel-app.com)
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Register Routes
# All endpoints will be prefixed with /api/v1 (Good API versioning practice)
app.include_router(api_router, prefix="/api/v1", tags=["Retrieval"])

# 4. Lambda Handler
# This 'handler' variable is what AWS Lambda looks for in Phase 1 Deployment
handler = Mangum(app)

# 5. Health Check (Root)
@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "active",
        "service": "HealthSage.AI Backend",
        "environment": "dev",  # You could pull this from Config
        "version": "0.1.0"
    }

if __name__ == "__main__":
    # For local debugging only
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)