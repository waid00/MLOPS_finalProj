import pytest
import sys
import platform
import ctypes
import logging
from importlib.util import find_spec
from pathlib import Path
from fastapi.testclient import TestClient

# Configure Logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- WINDOWS DLL FIX (Pathlib Compliant) ---
if platform.system() == "Windows":
    try:
        spec = find_spec("torch")
        if spec and spec.origin:
            # Use pathlib instead of os.path
            dll_path = Path(spec.origin).parent / "lib" / "c10.dll"
            if dll_path.exists():
                ctypes.CDLL(str(dll_path))
    except Exception as e:
        logger.warning(f"Attempted to pre-load c10.dll but failed: {e}")
# -------------------------------------------

# --- PATH FIX (Pathlib Compliant) ---
# Use pathlib to robustly find directories relative to this file
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
backend_dir = root_dir / "app" / "backend"

# Convert to string for sys.path as it expects strings
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))
# ------------------------------------

from app.backend.main import app

@pytest.fixture(scope="module")
def client():
    """
    Pytest fixture that initializes the TestClient.
    Uses context manager to ensure startup events (model loading) trigger correctly.
    """
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """
    Verifies the API is running and that the model has been loaded into memory.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["model_loaded"] is True

def test_predict_coding_override(client):
    """
    Verifies that the hybrid system correctly forces 'Coding & Development' 
    for explicit Python prompts, regardless of ML confidence.
    """
    response = client.post("/predict", json={"text": "Write a python script to calculate pi"})
    
    assert response.status_code == 200
    data = response.json()
    assert "topic_id" in data
    assert data["topic_label"] == "Coding & Development"
    assert data["topic_prob"] >= 0.95

def test_predict_creative_writing_override(client):
    """
    Verifies that the hybrid system correctly forces 'Creative Writing' 
    for poem/haiku requests.
    """
    response = client.post("/predict", json={"text": "Write a haiku about the ocean"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["topic_label"] == "Creative Writing"

def test_predict_empty(client):
    """
    Verifies the API handles empty string input gracefully without 500/503 errors.
    """
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 200 
    data = response.json()
    assert data["topic_label"] == "Uncategorized / Noise"

def test_predict_invalid_json(client):
    """
    Verifies that Pydantic validation correctly rejects malformed requests.
    """
    response = client.post("/predict", json={"wrong_field": "text"})
    assert response.status_code == 422