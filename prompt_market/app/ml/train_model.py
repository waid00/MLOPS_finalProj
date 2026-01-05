import pytest
import sys
import os
import platform
import ctypes
from importlib.util import find_spec
from fastapi.testclient import TestClient

# --- WINDOWS DLL FIX START ---
if platform.system() == "Windows":
    try:
        if (spec := find_spec("torch")) and spec.origin:
            dll_path = os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
            if os.path.exists(dll_path):
                ctypes.CDLL(os.path.normpath(dll_path))
    except Exception as e:
        print(f"Warning: Attempted to pre-load c10.dll but failed: {e}")
# --- WINDOWS DLL FIX END ---

# --- PATH FIX START ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../..'))
backend_dir = os.path.join(root_dir, 'app', 'backend')

if root_dir not in sys.path:
    sys.path.append(root_dir)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)
# --- PATH FIX END ---

from app.backend.main import app

# FIX: Use a fixture for the client to guarantee startup events (model loading) run
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """
    Verifies the API is running and reports model status.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["model_loaded"] is True

def test_predict_coding_override(client):
    """
    Tests the 'Safety Net' logic.
    """
    response = client.post("/predict", json={"text": "Write a python script to calculate pi"})
    
    assert response.status_code == 200
    data = response.json()
    assert "topic_id" in data
    assert data["topic_label"] == "Coding & Development"
    assert data["topic_prob"] >= 0.95

def test_predict_creative_writing_override(client):
    """
    Tests another category to ensure the expanded keyword list works.
    """
    response = client.post("/predict", json={"text": "Write a haiku about the ocean"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["topic_label"] == "Creative Writing"

def test_predict_empty(client):
    """
    Verifies the API handles empty input without crashing.
    """
    response = client.post("/predict", json={"text": ""})
    # Should return 200 now that we handle empty strings gracefully
    assert response.status_code == 200 
    data = response.json()
    # Expect our default "Uncategorized" response
    assert data["topic_label"] == "Uncategorized / Noise"

def test_predict_invalid_json(client):
    """
    Verifies Pydantic validation catches bad inputs.
    """
    response = client.post("/predict", json={"wrong_field": "text"})
    assert response.status_code == 422