import pytest
import sys
import os
import platform
import ctypes
from importlib.util import find_spec
from fastapi.testclient import TestClient

# --- WINDOWS DLL FIX START ---
# This must run BEFORE importing app.backend.main to fix [WinError 1114]
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
# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../..'))
backend_dir = os.path.join(root_dir, 'app', 'backend')

# Add root directory to path
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Crucial: Add backend directory to path so 'import model' inside main.py works
if backend_dir not in sys.path:
    sys.path.append(backend_dir)
# --- PATH FIX END ---

from app.backend.main import app

client = TestClient(app)

def test_health_check():
    """
    Verifies the API is running and reports model status.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_predict_coding_override():
    """
    Tests the 'Safety Net' logic. 
    Input 'Write a python script' should force 'Coding & Development' 
    regardless of the underlying ML cluster.
    """
    response = client.post("/predict", json={"text": "Write a python script to calculate pi"})
    
    # If model isn't loaded (e.g. in CI without artifact), we expect 503.
    # If loaded, we expect 200 and specific correct classification.
    if response.status_code == 200:
        data = response.json()
        assert "topic_id" in data
        assert data["topic_label"] == "Coding & Development"
        # We expect high confidence due to the override logic
        assert data["topic_prob"] >= 0.95
    else:
        assert response.status_code == 503

def test_predict_creative_writing_override():
    """
    Tests another category to ensure the expanded keyword list works.
    Input 'Write a poem' should trigger 'Creative Writing'.
    """
    response = client.post("/predict", json={"text": "Write a haiku about the ocean"})
    
    if response.status_code == 200:
        data = response.json()
        assert data["topic_label"] == "Creative Writing"
    else:
        assert response.status_code == 503

def test_predict_empty():
    """
    Verifies the API handles empty input without crashing.
    """
    response = client.post("/predict", json={"text": ""})
    # The API allows empty text, but the model might return noise or Uncategorized
    assert response.status_code == 200 
    data = response.json()
    assert "topic_label" in data

def test_predict_invalid_json():
    """
    Verifies Pydantic validation catches bad inputs.
    """
    response = client.post("/predict", json={"wrong_field": "text"})
    assert response.status_code == 422