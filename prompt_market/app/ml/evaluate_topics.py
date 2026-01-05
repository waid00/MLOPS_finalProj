from bertopic import BERTopic
import pandas as pd
from pathlib import Path

MODEL_PATH = "app/backend/bertopic_model.pkl"

def evaluate():
    try:
        topic_model = BERTopic.load(MODEL_PATH)
    except Exception as e:
        print(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
        return

    freq = topic_model.get_topic_info()
    print("Top 10 Topics found:")
    print(freq.head(12))
    
    print("\nTopic Representations (Top 5 words):")
    for i in range(8): # Check first 8
        try:
            topic_words = topic_model.get_topic(i)
            if isinstance(topic_words, list) and topic_words:
                print(f"Topic {i}: {topic_words[:5]}")
        except:
            pass

if __name__ == "__main__":
    evaluate()