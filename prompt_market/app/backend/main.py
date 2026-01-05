from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import uvicorn
import logging
from model import TopicModelClassifier

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Model Global
classifier = TopicModelClassifier()

# REPLACE: @app.on_event("startup") is deprecated. Use lifespan instead.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    classifier.load_model()
    yield
    # Shutdown: Clean up if needed (nothing to do here)

# Init App with lifespan
app = FastAPI(title="PromptMarket Topic API", version="1.0", lifespan=lifespan)

class PromptInput(BaseModel):
    text: str

class TopicResponse(BaseModel):
    topic_id: int
    topic_label: str
    topic_words: List[str]
    topic_prob: float

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": classifier.model is not None}

@app.post("/predict", response_model=TopicResponse)
def predict_topic(input_data: PromptInput):
    # SELF-HEALING FIX:
    # If the model isn't loaded (e.g., test runner weirdness), try loading it now.
    # This prevents the 503 error in your tests.
    if not classifier.model:
        logger.warning("Model not loaded in memory. Attempting lazy load...")
        classifier.load_model()

    if not classifier.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = classifier.predict(input_data.text)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)