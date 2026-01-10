import logging
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import TopicModelClassifier

# --- 1. KONFIGURACE LOGOVÁNÍ (Místo print()) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Globální instance modelu
classifier = TopicModelClassifier()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Obsluhuje životní cyklus aplikace (startup/shutdown).
    Zajišťuje načtení ML modelu do paměti při startu.
    """
    logger.info("Spouštím backend: Načítám model...")
    classifier.load_model()
    yield
    logger.info("Vypínám backend: Uklízím zdroje...")

# Inicializace aplikace
app = FastAPI(
    title="PromptMarket Topic API",
    version="1.0",
    description="API pro klasifikaci témat pomocí BERTopic modelu.",
    lifespan=lifespan
)

# --- DATOVÉ MODELY (Pydantic) ---
class PromptInput(BaseModel):
    text: str

class TopicResponse(BaseModel):
    topic_id: int
    topic_label: str
    topic_words: List[str]
    topic_prob: float

# --- ENDPOINTY ---

@app.get("/health")
def health_check():
    """
    Kontroluje stav aplikace a načtení modelu.

    Returns:
        dict: Status aplikace a flag indikující načtení modelu.
    """
    return {
        "status": "healthy",
        "model_loaded": classifier.model is not None
    }

@app.post("/predict", response_model=TopicResponse)
def predict_topic(input_data: PromptInput):
    """
    Přijme textový vstup a predikuje jeho téma.
    
    Využívá hybridní přístup (ML model + klíčová slova) definovaný
    v třídě TopicModelClassifier.

    Args:
        input_data (PromptInput): JSON obsahující klíč 'text'.

    Returns:
        TopicResponse: Strukturovaná odpověď s ID tématu, labelem a pravděpodobností.

    Raises:
        HTTPException (503): Pokud model není načten.
        HTTPException (500): Při chybě predikce.
    """
    # Self-healing: Pokus o lazy-load, pokud model v paměti chybí
    if not classifier.model:
        logger.warning("Model nebyl v paměti. Pokouším se o lazy-load...")
        classifier.load_model()

    if not classifier.model:
        logger.error("Predikce selhala: Model se nepodařilo načíst.")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Zpracovávám predikci pro text (délka: {len(input_data.text)})")
        result = classifier.predict(input_data.text)
        return result
    except Exception as e:
        logger.error(f"Chyba při predikci: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)