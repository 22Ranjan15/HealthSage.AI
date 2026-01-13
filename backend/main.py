from fastapi import FastAPI
from mangum import Mangum

app = FastAPI(title="HealthSage.AI")
handler = Mangum(app)

@app.get("/")
def health_check():
    return {
        "status": "Healthy",
        "service": "HealthSage.AI Backend",
        "version": "0.0.1"
    }

@app.get("/test")
def test_route():
    return {"message": "If you see this, FastAPI is working perfectly!"}