import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Configure logging
logger = logging.getLogger(__name__)

class TopicModelClassifier:
    """
    A wrapper class for the BERTopic model that provides hybrid classification capabilities.
    
    It combines the unsupervised ML model with a rule-based keyword override system
    to ensure high precision for specific business categories (HR Insights).
    """

    def __init__(self):
        """
        Initialize the classifier configuration and keyword maps.
        Does not load the heavy model into memory until load_model() is called.
        """
        self.model = None
        # Use absolute path relative to this file to find the model artifact
        self.model_path = Path(__file__).parent / "bertopic_model.pkl"
        
        self.default_label = "General Work Task"

        # ---------------------------------------------------------
        # STRATEGY: EMPLOYEE INSIGHT ENGINE (8 KEY CATEGORIES)
        # ---------------------------------------------------------
        self.keyword_overrides = {
            # 1. TECHNICAL SKILLS (For Developers & Engineers - Skill Gap Analysis)
            "Technical & Coding": [
                "python", "javascript", "code", "script", "function", "java ", "c++", 
                "sql", "api", "bug", "terminal", "bash", "git", "regex", "json",
                "docker", "kubernetes", "deploy", "stack trace", "debug"
            ],
            
            # 2. HR & EMPLOYMENT POLICIES (What employees ask about their job/contract)
            "HR & Employment Policies": [
                "holiday", "vacation", "leave", "sick day", "benefits", "insurance",
                "promotion", "hiring", "onboarding", "interview", "salary", "bonus",
                "career path", "resignation", "notice period", "work from home", "remote",
                "paternity", "maternity", "contract"
            ],
            
            # 3. EMPLOYEE WELLBEING (CRITICAL - Signals for HR Intervention)
            "Employee Wellbeing (Flagged)": [
                "burnout", "stress", "tired", "overwhelmed", "anxiety", "mental health",
                "doctor", "therapist", "motivation", "conflict", "harassment",
                "unhappy", "quit", "toxic", "workload", "exhausted", "depression"
            ],
            
            # 4. IT SUPPORT & INFRASTRUCTURE (Internal technical issues)
            "IT Support & Infrastructure": [
                "wifi", "vpn", "password", "login", "cant access", "printer", "laptop",
                "screen", "mouse", "keyboard", "broken", "install", "update", "connection",
                "server down", "access denied", "2fa", "authentication"
            ],

            # 5. LEGAL & COMPLIANCE (Corporate Governance queries)
            "Legal & Compliance": [
                "nda", "gdpr", "policy", "regulation", "law", "legal",
                "compliance", "audit", "agreement", "term sheet", "intellectual property",
                "privacy", "data protection"
            ],

            # 6. FINANCE & BUDGETING (Admin & Expensing)
            "Finance & Budgeting": [
                "invoice", "budget", "cost", "expense", "tax", "vat", "price", 
                "quote", "billing", "payment", "reimbursement", "receipt", "po number"
            ],

            # 7. LEARNING & DEVELOPMENT (Upskilling trends)
            "Learning & Development": [
                "how to", "tutorial", "explain", "learn", "course", "training", 
                "certification", "study", "what is", "basics of", "best practice",
                "guide", "manual", "documentation"
            ],

            # 8. COMMUNICATION & TRANSLATION (General productivity tools)
            "Communication & Translation": [
                "translate", "english", "german", "french", "spanish", "proofread",
                "grammar", "rewrite", "email", "draft", "summary", "paraphrase",
                "spell check", "tone", "formal", "presentation", "slide", "excel", "spreadsheet"
            ]
        }
        
        # Fallback mapping from BERTopic IDs (if model is used directly)
        self.business_categories = {
             0: "General Assistance",
             1: "Technical Support",
             2: "Content Generation"
        }

    def load_model(self) -> None:
        """
        Loads the BERTopic artifact and the SentenceTransformer embedding model from disk.
        """
        try:
            if self.model_path.exists():
                logger.info(f"Loading model from {self.model_path}")
                # We use 'all-mpnet-base-v2' to match the training script configuration
                embedding_model = SentenceTransformer("all-mpnet-base-v2") 
                self.model = BERTopic.load(str(self.model_path), embedding_model=embedding_model)
                logger.info("Model loaded successfully.")
            else:
                logger.error(f"Model file not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def _map_topic_to_business_label(self, topic_id: int) -> str:
        """
        Maps an internal BERTopic integer ID to a human-readable business category.
        """
        if topic_id == -1:
            return "Uncategorized / Noise"
        
        if not self.model:
            return self.default_label
        
        # 1. Manual mapping (if defined)
        if topic_id in self.business_categories:
            return self.business_categories[topic_id]

        # 2. Get info from model
        try:
            info = self.model.get_topic_info(topic_id)
            if info.empty:
                return self.default_label
            topic_name = info['Name'].values[0]
            clean_name = topic_name.lower()
        except:
            return f"Topic {topic_id}"
        
        # 3. Try to match keywords in the Topic Name itself
        for category, keywords in self.keyword_overrides.items():
            if any(k in clean_name for k in keywords):
                return category

        # 4. Fallback: Return a clean version of the topic words
        parts = clean_name.split('_')
        if len(parts) > 1:
            return f"Topic: {', '.join(parts[1:4])}"
        
        return f"Topic {topic_id}"

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predicts the topic using Hybrid approach (Rules > Model).
        
        Steps:
        1. Run ML Model to get base prediction.
        2. Run Keyword Override to force specific categories if keywords are present.
        """
        if not self.model:
            raise ValueError("Model not initialized")
        
        if not text or not text.strip():
             return {
                "topic_id": -1,
                "topic_label": "Uncategorized / Noise",
                "topic_words": [],
                "topic_prob": 0.0
             }

        # --- STEP 1: Standard ML Prediction ---
        topics, probs = self.model.transform([text])
        topic_id = topics[0]
        
        confidence = 0.0
        if probs is not None:
            if isinstance(probs, np.ndarray) and probs.size > 0:
                # Handle potential variations in probability output shape
                confidence = float(np.max(probs))

        topic_words = []
        if self.model and topic_id != -1:
            try:
                topic_data = self.model.get_topic(topic_id)
                if isinstance(topic_data, (list, tuple)) and topic_data:
                    topic_words = [word[0] for word in topic_data if isinstance(word, (list, tuple)) and len(word) > 0]
            except Exception:
                pass
        
        # Get default label from ML
        final_label = self._map_topic_to_business_label(topic_id)

        # --- STEP 2: KEYWORD OVERRIDE (The Hybrid Logic) ---
        # If a keyword is found, we override the ML label and boost confidence.
        text_lower = text.lower()
        for category, keywords in self.keyword_overrides.items():
            if any(kw in text_lower for kw in keywords):
                final_label = category
                confidence = max(confidence, 0.95) # High confidence for rule-based matches
                break
        
        return {
            "topic_id": int(topic_id),
            "topic_label": final_label,
            "topic_words": topic_words,
            "topic_prob": confidence
        }