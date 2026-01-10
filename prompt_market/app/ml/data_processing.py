import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_URL = "hf://datasets/fka/awesome-chatgpt-prompts/prompts.csv"

def load_data(path_or_url: str) -> pd.DataFrame:
    """
    Loads dataset from a local path or HuggingFace URL.
    """
    try:
        logger.info(f"Loading data from {path_or_url}...")
        df = pd.read_csv(path_or_url)
        logger.info(f"Successfully loaded {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    1. Lowercase
    2. Remove excessive whitespace
    3. Remove specific 'act as' prefixes if they dominate clustering (optional, kept minimal here)
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove specific noise (optional based on EDA)
    # text = text.replace("i want you to act as", "") 
    
    # Remove excessive whitespace/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies cleaning, filtering, and feature engineering.
    
    Decisions:
    1. Drop Duplicates: Exact matches skew cluster weights.
    2. Min Length: Prompts < 15 chars are usually noise or "hi", lacking semantic context.
    3. Feature Eng: Added word_count for potential filtering/analysis.
    """
    initial_count = len(df)
    
    # 1. Drop duplicates
    df = df.drop_duplicates(subset=['prompt'])
    
    # 2. Handle missing
    df = df.dropna(subset=['prompt'])
    
    # 3. Feature Engineering
    df['cleaned_prompt'] = df['prompt'].apply(clean_text)
    df['word_count'] = df['cleaned_prompt'].apply(lambda x: len(x.split()))
    
    # 4. Filtering
    # Threshold Justification: < 4 words is rarely a "prompt" in this context.
    df = df[df['word_count'] >= 4]
    
    logger.info(f"Preprocessing complete. Dropped {initial_count - len(df)} rows. Final count: {len(df)}")
    return df

def get_eda_stats(df: pd.DataFrame) -> dict:
    """Returns basic EDA statistics."""
    stats = {
        "avg_len": df['word_count'].mean(),
        "min_len": df['word_count'].min(),
        "max_len": df['word_count'].max(),
        "top_act_prefixes": df['act'].value_counts().head(5).to_dict() if 'act' in df.columns else {}
    }
    return stats

if __name__ == "__main__":
    # For testing the script independently
    df = load_data(DATA_URL)
    df_clean = preprocess_data(df)
    logger.info(f"EDA Stats: {get_eda_stats(df_clean)}")
    
    # Save for reproducible training
    output_path = Path("processed_prompts.csv")
    df_clean.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")