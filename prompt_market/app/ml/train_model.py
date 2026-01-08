import platform
import ctypes
import logging
import argparse
from importlib.util import find_spec
from pathlib import Path
from typing import Any, List, Optional, Union, cast

# Third-party imports
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

# Local imports
# Assuming data_processing.py is in the same directory or Python path
try:
    from data_processing import load_data, preprocess_data, DATA_URL
except ImportError:
    # Fallback for the sake of the script running standalone if dependencies are missing
    logging.warning("Could not import data_processing. Ensure it exists.")
    DATA_URL = "placeholder_url"
    # Adjusted parameter name to 'path_or_url' to match the inferred signature of the real function
    def load_data(path_or_url): return pd.DataFrame({'cleaned_prompt': []})
    def preprocess_data(df): return df

# --- Configuration ---
# Define paths relative to this script's location to avoid CWD errors
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "app" / "backend" / "bertopic_model.pkl"
DEFAULT_DATA_SAVE_PATH = BASE_DIR / "app" / "backend" / "labeled_data.csv"

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- WINDOWS DLL FIX ---
def apply_windows_patch():
    """
    Fixes the missing c10.dll error often found on Windows environments with Torch.
    """
    if platform.system() == "Windows":
        try:
            spec = find_spec("torch")
            if spec and spec.origin:
                dll_path = Path(spec.origin).parent / "lib" / "c10.dll"
                if dll_path.exists():
                    ctypes.CDLL(str(dll_path))
                    logger.debug("Successfully pre-loaded c10.dll")
        except Exception as e:
            logger.warning(f"Attempted to pre-load c10.dll but failed: {e}")

# Apply patch immediately
apply_windows_patch()
# -----------------------

def train_and_save(
    data_url: str = DATA_URL,
    model_path: Path = DEFAULT_MODEL_PATH,
    data_save_path: Path = DEFAULT_DATA_SAVE_PATH
) -> None:
    """
    Orchestrates the training pipeline for the BERTopic model.

    Args:
        data_url: URL or path to the source data.
        model_path: Path where the trained model will be saved.
        data_save_path: Path where the labeled dataset will be saved.
    """
    # 1. Load & Process
    logger.info(f"Loading data from {data_url}...")
    try:
        df = load_data(data_url)
        df = preprocess_data(df)
    except Exception as e:
        logger.error(f"Failed to load or process data: {e}")
        return

    if 'cleaned_prompt' not in df.columns:
        logger.error("Dataframe missing required column 'cleaned_prompt'")
        return

    docs = df['cleaned_prompt'].tolist()
    logger.info(f"Training on {len(docs)} documents.")

    # 2. Configuration
    logger.info("Initializing model components...")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Enforce numpy output directly to avoid manual type checking later
    # Explicitly casting to np.ndarray to satisfy type checkers since convert_to_numpy=True guarantees it
    embeddings = cast(np.ndarray, embedding_model.encode(docs, show_progress_bar=True, convert_to_numpy=True))

    # UMAP: Reduce dimensions of embeddings
    umap_model = UMAP(
        n_neighbors=15, 
        n_components=5, 
        min_dist=0.0, 
        metric='cosine', 
        random_state=42
    )

    # HDBSCAN: Cluster the reduced embeddings
    # prediction_data=True is crucial for BERTopic.transform() later
    hdbscan_model = HDBSCAN(
        min_cluster_size=7, 
        metric='euclidean', 
        cluster_selection_method='eom', 
        prediction_data=True
    )

    # Custom Stop Words
    custom_stop_words = [
        'write', 'want', 'act', 'reply', 'create', 'generate', 'make', 
        'tell', 'ask', 'explain', 'prompt', 'chatgpt', 'ai'
    ]
    # Use explicit conversion to list to avoid set/list mismatch in older sklearn versions
    final_stop_words = list(text.ENGLISH_STOP_WORDS.union(custom_stop_words))

    vectorizer_model = CountVectorizer(stop_words=final_stop_words, ngram_range=(1, 2))

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=True
    )

    # 3. Fit
    logger.info("Fitting BERTopic model...")
    topics, probs = topic_model.fit_transform(docs, embeddings)

    # 4. Save Artifacts
    logger.info(f"Saving model artifact to {model_path}...")

    # Ensure parent directory exists
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)

    # Safetensors is faster and safer than pickle, but requires `pip install safetensors`
    try:
        topic_model.save(str(model_path), serialization="safetensors", save_embedding_model=True)
    except ImportError:
        logger.warning("Safetensors not found. Falling back to pickle/pytorch default.")
        topic_model.save(str(model_path), save_embedding_model=True)
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return

    # Save Data with topics for eval
    logger.info(f"Saving labeled data to {data_save_path}...")
    if not data_save_path.parent.exists():
        data_save_path.parent.mkdir(parents=True, exist_ok=True)
        
    df['topic'] = topics
    df.to_csv(data_save_path, index=False)
    
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERTopic model")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to save the model")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_SAVE_PATH, help="Path to save the labeled data")
    
    args = parser.parse_args()
    
    train_and_save(model_path=args.model_path, data_save_path=args.data_path)