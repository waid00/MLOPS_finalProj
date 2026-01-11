import platform
import ctypes
import logging
import os
import sys
from pathlib import Path
from importlib.util import find_spec

# --- 1. WINDOWS DLL PATCH (CLEAN CODE & ROBUSTNESS) ---
def apply_windows_torch_patch():
    """
    Fixes OSError: [WinError 1114] / c10.dll on Windows.
    Must be called before importing bertopic or torch.
    """
    if platform.system() == "Windows":
        logging.info("Applying Windows DLL patch for PyTorch...")
        try:
            spec = find_spec("torch")
            if spec and spec.origin:
                torch_lib_path = Path(spec.origin).parent / "lib"
                if torch_lib_path.exists():
                    os.add_dll_directory(str(torch_lib_path))
                    ctypes.CDLL(str(torch_lib_path / "c10.dll"))
                    logging.info(f"Successfully patched DLL paths from {torch_lib_path}")
        except Exception as e:
            logging.warning(f"Windows DLL patch failed (might not be needed): {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

apply_windows_torch_patch()

# --- 2. IMPORTY ---
import mlflow
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

try:
    from data_processing import load_data, preprocess_data, DATA_URL
except ImportError:
    logger.error("Data processing module not found. Check your file structure.")
    sys.exit(1)

def run_experiments():
    """
    Runs a systematic hyperparameter grid search for the Employee Insight Engine.
    Tracks n_topics and outlier_ratio using MLflow.
    """
    logger.info(f"Loading data from {DATA_URL}")
    df = load_data(DATA_URL)
    df = preprocess_data(df)
    docs = df['cleaned_prompt'].tolist()
    
    mlflow.set_experiment("Employee_Insight_Optimization")

    # Matice experimentů - Rozšířeno na 10 běhů pro hlubší analýzu
    experiments = [
        # --- PŮVODNÍCH 5 ---
        {"umap_neighbors": 15, "min_cluster_size": 5,  "min_samples": 5, "note": "Baseline - High Granularity"},
        {"umap_neighbors": 15, "min_cluster_size": 10, "min_samples": 5, "note": "Target Configuration"},
        {"umap_neighbors": 15, "min_cluster_size": 20, "min_samples": 5, "note": "Strict Trend Detection"},
        {"umap_neighbors": 5,  "min_cluster_size": 10, "min_samples": 5, "note": "Local Structure Focus"},
        {"umap_neighbors": 30, "min_cluster_size": 10, "note": "Global Semantic Focus"},
        
        # --- NOVÝCH 5 (SYSTEMATICKÉ TESTOVÁNÍ EXTRÉMŮ) ---
        {"umap_neighbors": 15, "min_cluster_size": 3,  "min_samples": 1, "note": "Overfitting Test (Noise Sensitivity)"},
        {"umap_neighbors": 15, "min_cluster_size": 40, "note": "Underfitting Test (Only Massive Trends)"},
        {"umap_neighbors": 50, "min_cluster_size": 10, "note": "Ultra-Global Manifold Approximation"},
        {"umap_neighbors": 15, "min_cluster_size": 10, "min_samples": 1, "note": "Low Noise Filtering (Min Samples 1)"},
        {"umap_neighbors": 15, "min_cluster_size": 10, "min_samples": 15, "note": "Aggressive Noise Filtering (Min Samples 15)"}
    ]

    model_name = "all-mpnet-base-v2"
    logger.info(f"Generating embeddings using {model_name}...")
    embedding_model = SentenceTransformer(model_name)
    embeddings = np.asarray(embedding_model.encode(docs, show_progress_bar=True))

    for i, params in enumerate(experiments):
        run_name = f"Exp_{i}_{params['note'][:25]}"
        
        with mlflow.start_run(run_name=run_name):
            logger.info(f"Starting experiment: {run_name}")
            
            umap_model = UMAP(
                n_neighbors=params['umap_neighbors'], 
                n_components=5, 
                min_dist=0.0, 
                metric='cosine', 
                random_state=42
            )
            
            # min_samples ovlivňuje, jak moc model "penalizuje" šum
            hdbscan_model = HDBSCAN(
                min_cluster_size=params['min_cluster_size'],
                min_samples=params.get('min_samples', params['min_cluster_size']), 
                metric='euclidean', 
                cluster_selection_method='eom', 
                prediction_data=True
            )
            
            vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))
            
            topic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                calculate_probabilities=True,
                verbose=False
            )
            
            topics, _ = topic_model.fit_transform(docs, embeddings)
            
            topic_info = topic_model.get_topic_info()
            n_topics = len(topic_info) - 1
            outlier_count = topic_info.loc[topic_info['Topic'] == -1, 'Count'].sum() if -1 in topic_info['Topic'].values else 0
            outlier_ratio = outlier_count / len(docs)
            
            mlflow.log_params({
                "umap_neighbors": params['umap_neighbors'],
                "min_cluster_size": params['min_cluster_size'],
                "min_samples": params.get('min_samples', params['min_cluster_size']),
                "embedding_model": model_name,
                "note": params['note']
            })
            mlflow.log_metric("n_topics", n_topics)
            mlflow.log_metric("outlier_ratio", outlier_ratio)
            
            summary = topic_info.head(15)[['Topic', 'Count', 'Name']].to_string()
            mlflow.log_text(summary, "topic_summary.txt")

            logger.info(f"Finished {run_name}: {n_topics} topics, {outlier_ratio:.2%} outliers.")

if __name__ == "__main__":
    run_experiments()