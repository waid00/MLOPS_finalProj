import platform
import ctypes
from importlib.util import find_spec
from pathlib import Path

# --- WINDOWS DLL FIX (MUSÍ BÝT PRVNÍ) ---
if platform.system() == "Windows":
    try:
        spec = find_spec("torch")
        if spec and spec.origin:
            dll_path = Path(spec.origin).parent / "lib" / "c10.dll"
            if dll_path.exists():
                ctypes.CDLL(str(dll_path))
                print("Successfully pre-loaded c10.dll for evaluate_topics")
    except Exception as e:
        print(f"Warning: Failed to pre-load c10.dll: {e}")
# ----------------------------------------

from bertopic import BERTopic
import pandas as pd
from typing import Any

# Použití absolutní cesty pro jistotu
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # prompt_market/
MODEL_PATH = BASE_DIR / "app" / "backend" / "bertopic_model.pkl"

def evaluate():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        topic_model = BERTopic.load(str(MODEL_PATH))
    except Exception as e:
        print(f"Model not found at {MODEL_PATH}. Run train_model.py first. Error: {e}")
        return

    freq = topic_model.get_topic_info()
    print("\nTop 10 Topics found:")
    print(freq.head(12))
    
    print("\nTopic Representations (Top 5 words):")
    # Vypíšeme prvních pár témat
    for i in range(min(10, len(freq) - 1)):
        topic_words: Any = topic_model.get_topic(i)
        words = []
        if isinstance(topic_words, (list, tuple)):
            try:
                words = [w for (w, *_) in topic_words[:5] if isinstance(w, str)]
            except Exception:
                words = []
        elif isinstance(topic_words, dict):
            try:
                words = list(topic_words.keys())[:5]
            except Exception:
                words = []
        if words:
            print(f"Topic {i}: {words}")

if __name__ == "__main__":
    evaluate()