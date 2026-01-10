import platform
import ctypes
import logging
from importlib.util import find_spec
from pathlib import Path

# --- WINDOWS DLL FIX ---
# This must be run BEFORE importing bertopic or torch
if platform.system() == "Windows":
    try:
        spec = find_spec("torch")
        if spec and spec.origin:
            dll_path = Path(spec.origin).parent / "lib" / "c10.dll"
            if dll_path.exists():
                ctypes.CDLL(str(dll_path))
                print("Successfully pre-loaded c10.dll for Windows")
    except Exception as e:
        print(f"Warning: Failed to pre-load c10.dll: {e}")
# -----------------------



import mlflow
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from data_processing import load_data, preprocess_data, DATA_URL
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_experiments():
    """
    Runs a grid of experiments to determine optimal BERTopic hyperparameters.
    """
    # Load Data
    df = load_data(DATA_URL)
    df = preprocess_data(df)
    docs = df['cleaned_prompt'].tolist()
    
    mlflow.set_experiment("BERTopic_ChatGPT_Prompts")

    # Experiment Grid
    # Rationale: Varying n_neighbors affects local vs global structure in UMAP.
    # Varying min_cluster_size affects how granular the topics are.
    experiments = [
        {"umap_neighbors": 15, "min_cluster_size": 5, "embedding": "all-MiniLM-L6-v2"},
        {"umap_neighbors": 15, "min_cluster_size": 10, "embedding": "all-MiniLM-L6-v2"}, # Likely balanced
        {"umap_neighbors": 5,  "min_cluster_size": 5,  "embedding": "all-MiniLM-L6-v2"}, # Local focus
        {"umap_neighbors": 30, "min_cluster_size": 15, "embedding": "all-MiniLM-L6-v2"}, # Global focus
    ]

    for i, params in enumerate(experiments):
        run_name = f"Run_{i}_n{params['umap_neighbors']}_sz{params['min_cluster_size']}"
        
        with mlflow.start_run(run_name=run_name):
            logger.info(f"Starting {run_name}")
            
            # 1. Embeddings
            embedding_model = SentenceTransformer(params['embedding'])
            embeddings = embedding_model.encode(docs, show_progress_bar=False)
            # Convert to numpy array safely - handles Tensor, list, and ndarray
            import numpy as np
            embeddings = np.asarray(embeddings)
            
            # 2. Dimensionality Reduction
            umap_model = UMAP(n_neighbors=params['umap_neighbors'], 
                              n_components=5, 
                              min_dist=0.0, 
                              metric='cosine', 
                              random_state=42)
            
            # 3. Clustering
            hdbscan_model = HDBSCAN(min_cluster_size=params['min_cluster_size'], 
                                    metric='euclidean', 
                                    cluster_selection_method='eom', 
                                    prediction_data=True)
            
            # 4. Vectorizer (removes stopwords)
            vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))
            
            # Train
            topic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                calculate_probabilities=True,
                verbose=True
            )
            topics, probs = topic_model.fit_transform(docs, embeddings)
            
            # Metrics
            topic_info = topic_model.get_topic_info()
            n_topics = len(topic_info) - 1 # Exclude -1 outliers
            outlier_count = topic_info.loc[topic_info['Topic'] == -1, 'Count'].sum() if -1 in topic_info['Topic'].values else 0
            outlier_ratio = outlier_count / len(docs)
            
            # Coherence (Proxy: using c_v is slow, using semantic diversity as proxy here or external lib)
            # For this script, we log diversity of topic representations
            
            # Logging
            mlflow.log_params(params)
            mlflow.log_metric("n_topics", n_topics)
            mlflow.log_metric("outlier_ratio", outlier_ratio)
            
            # Log top words for inspection
            top_topics = topic_model.get_topic(0)
            if top_topics:
                mlflow.log_text(str(top_topics), "top_topic_0_words.txt")

            logger.info(f"Finished {run_name}: Found {n_topics} topics.")

if __name__ == "__main__":
    run_experiments()