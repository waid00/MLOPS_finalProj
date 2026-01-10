import platform
import ctypes
import logging
import argparse
from importlib.util import find_spec
from pathlib import Path
from typing import Any, List, Optional, Union, cast

# --- WINDOWS DLL FIX (MUSÍ BÝT PŘED IMPORTEM TORCH/BERTOPIC) ---
def apply_windows_patch():
    """Fixes the missing c10.dll error often found on Windows."""
    if platform.system() == "Windows":
        try:
            spec = find_spec("torch")
            if spec and spec.origin:
                dll_path = Path(spec.origin).parent / "lib" / "c10.dll"
                if dll_path.exists():
                    ctypes.CDLL(str(dll_path))
                    logging.info("Successfully pre-loaded c10.dll")
        except Exception as e:
            logging.warning(f"Warning: Failed to pre-load c10.dll: {e}")

apply_windows_patch()
# --------------------------------------------------------------

# NYNÍ MŮŽEME BEZPEČNĚ IMPORTOVAT ZBYTEK
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

# Local imports
try:
    from data_processing import load_data, preprocess_data, DATA_URL
except ImportError:
    logging.warning("Could not import data_processing. Ensure it exists.")
    DATA_URL = "placeholder_url"
    def load_data(path_or_url): return pd.DataFrame({'cleaned_prompt': []})
    def preprocess_data(df): return df

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "app" / "backend" / "bertopic_model.pkl"
DEFAULT_DATA_SAVE_PATH = BASE_DIR / "app" / "backend" / "labeled_data.csv"

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_and_save(
    data_url: str = DATA_URL,
    model_path: Path = DEFAULT_MODEL_PATH,
    data_save_path: Path = DEFAULT_DATA_SAVE_PATH
) -> None:
    """Orchestrates the training pipeline for the BERTopic model."""
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

    # 1. BETTER EMBEDDINGS (Vylepšení: Silnější model)
    logger.info("Encoding embeddings (using all-mpnet-base-v2)...")
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = cast(np.ndarray, embedding_model.encode(docs, show_progress_bar=True, convert_to_numpy=True))

    # 2. TUNED HYPERPARAMETERS (Vylepšení: Méně šumu)
    umap_model = UMAP(
        n_neighbors=15, 
        n_components=5, 
        min_dist=0.0, 
        metric='cosine', 
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=7,
        min_samples=2,      # <-- VYLEPŠENÍ: Snižuje počet outliers (-1)
        metric='euclidean', 
        cluster_selection_method='eom', 
        prediction_data=True
    )

    # 3. BETTER REPRESENTATION (Vylepšení: Čistší klíčová slova)
    # KeyBERTInspired vybírá slova, která lépe reprezentují sémantiku tématu
    representation_model = KeyBERTInspired()

    # Custom Stop Words
    custom_stop_words = ['write', 'want', 'act', 'reply', 'create', 'generate', 'make', 'tell', 'ask', 'explain', 'prompt', 'chatgpt', 'ai']
    final_stop_words = list(text.ENGLISH_STOP_WORDS.union(custom_stop_words))
    vectorizer_model = CountVectorizer(stop_words=final_stop_words, ngram_range=(1, 2))

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model, # <-- Přidáno vylepšení
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=True
    )

    # 4. FIT
    logger.info("Fitting BERTopic model...")
    topics, probs = topic_model.fit_transform(docs, embeddings)

    # 5. POST-PROCESSING: REDUCE OUTLIERS (Vylepšení: Přiřazení nezařazených)
    logger.info(f"Original outlier count: {topics.count(-1)}")
    if topics.count(-1) > 0:
        logger.info("Reducing outliers using probabilistic assignment...")
        # Přiřadí body -1 k nejbližšímu tématu
        new_topics = topic_model.reduce_outliers(docs, topics)
        topic_model.update_topics(docs, topics=new_topics)
        topics = new_topics
        logger.info(f"New outlier count: {topics.count(-1)}")

    # 6. SAVE
    logger.info(f"Saving model artifact to {model_path}...")
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        topic_model.save(str(model_path), serialization="safetensors", save_embedding_model=True)
    except Exception as e:
        logger.warning(f"Safetensors save failed, using default: {e}")
        topic_model.save(str(model_path), save_embedding_model=True)

    logger.info(f"Saving labeled data to {data_save_path}...")
    if not data_save_path.parent.exists():
        data_save_path.parent.mkdir(parents=True, exist_ok=True)
        
    df['topic'] = topics
    df.to_csv(data_save_path, index=False)
    
    # Print info for user
    freq = topic_model.get_topic_info()
    logger.info(f"Training pipeline completed. Found {len(freq)-1} topics.")
    logger.info("Top 10 Topics:\n" + str(freq.head(10)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERTopic model")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_SAVE_PATH)
    args = parser.parse_args()
    train_and_save(model_path=args.model_path, data_save_path=args.data_path)