import logging
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class TopicModelClassifier:
    def __init__(self):
        self.model = None
        # FIX: Use absolute path relative to this file, not the command line CWD
        # This ensures tests run from root can still find the model in app/backend/
        self.model_path = Path(__file__).parent / "bertopic_model.pkl"
        
        # Business Label Mapping (Used for looking up topic IDs directly)
        self.business_categories = {
            "linux_terminal_code_php": "Coding & Development",
            "essay_write_title_article": "Content Creation",
            "act_interviewer_position_candidate": "Career & Roleplay",
            "translate_english_sentence_language": "Language & Translation",
            "story_character_plot_write": "Creative Writing",
            "seo_keywords_content_article": "Marketing & SEO",
            "excel_formula_sheet_data": "Productivity & Tools",
            "math_solve_equation_step": "Education & Academic",
            "mental_health_advice_life": "Health & Lifestyle",
            "socrates_philosophy_ethics": "Philosophy & Logic"
        }
        self.default_label = "General AI Assistant"

        # Keyword Overrides
        self.keyword_overrides = {
            "Coding & Development": [
                "python", "javascript", "code", "script", "function", "css", "html", "java ", 
                "c++", "sql", "api", "bug", "linux", "terminal", "bash", "git", "regex", "json"
            ],
            "Content Creation": [
                "essay", "blog", "article", "summary", "title", "outline", "draft", 
                "cover letter", "email", "rewrite", "paraphrase", "copywriting", "text"
            ],
            "Marketing & SEO": [
                "seo", "marketing", "ad copy", "social media", "instagram", "facebook", 
                "twitter", "linkedin", "keyword", "audience", "brand"
            ],
            "Creative Writing": [
                "story", "poem", "song", "lyrics", "haiku", "script", "plot", "character", 
                "novel", "fiction", "screenplay", "narrative", "rhyme"
            ],
            "Roleplay & Persona": [
                "act as", "pretend", "you are a", "simulate", "roleplay", "interviewer", 
                "candidate", "consultant", "expert", "therapist"
            ],
            "Language & Translation": [
                "translate", "english", "spanish", "french", "german", "japanese", 
                "language", "grammar", "correct", "proofread", "vocabulary"
            ],
            "Productivity & Tools": [
                "excel", "spreadsheet", "formula", "csv", "schedule", "plan", "organize", 
                "list", "table", "format", "meeting"
            ],
            "Education & Academic": [
                "teach", "explain", "student", "math", "physics", "history", "science", 
                "homework", "tutor", "quiz", "test", "learn"
            ],
            "Health & Lifestyle": [
                "diet", "food", "workout", "exercise", "recipe", "meal plan", "mental health", 
                "nutrition", "gym", "advice", "motivation"
            ],
            "Philosophy & Logic": [
                "socrates", "philosophy", "ethics", "logic", "stoic", "reason", "argument", 
                "debate", "fallacy", "critical thinking"
            ]
        }

    def load_model(self):
        try:
            if self.model_path.exists():
                logger.info(f"Loading model from {self.model_path}")
                # Explicitly load embedding model to avoid "No embedding model found" error
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.model = BERTopic.load(str(self.model_path), embedding_model=embedding_model)
                logger.info("Model loaded successfully.")
            else:
                logger.error(f"Model file not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def _map_topic_to_business_label(self, topic_id: int) -> str:
        if topic_id == -1:
            return "Uncategorized / Noise"
        
        if not self.model:
            return self.default_label
        
        info = self.model.get_topic_info(topic_id)
        if info.empty:
            return self.default_label
            
        topic_name = info['Name'].values[0]
        clean_name = topic_name.lower()
        
        # 1. Try to match keywords in the Topic Name itself
        for category, keywords in self.keyword_overrides.items():
            topic_keywords = clean_name.split('_')
            if any(k in topic_keywords for k in keywords) or any(k in clean_name for k in keywords):
                return category

        # 2. Handle generic "junk" topic names
        if any(x in clean_name for x in ['request', 'provide', 'need', 'help', 'ask', 'assistant']):
            return "General AI Assistance"
            
        # 3. Fallback
        readable_name = ", ".join(clean_name.split('_')[1:4])
        return f"Topic: {readable_name}"

    def predict(self, text: str):
        if not self.model:
            raise ValueError("Model not initialized")
        
        # FIX: Handle empty or whitespace-only strings immediately
        # This prevents the model from running transform() on empty input
        if not text or not text.strip():
             return {
                "topic_id": -1,
                "topic_label": "Uncategorized / Noise",
                "topic_words": [],
                "topic_prob": 0.0
             }

        # 1. Standard ML Prediction
        topics, probs = self.model.transform([text])
        topic_id = topics[0]
        
        confidence = 0.0
        if probs is not None:
            if isinstance(probs, np.ndarray) and probs.size > 0:
                if probs.ndim >= 2:
                    confidence = float(np.max(probs[0]))
                elif probs.ndim == 1:
                    confidence = float(np.max(probs))
                else:
                    confidence = float(probs.item())

        topic_words = []
        if self.model and topic_id != -1:
            try:
                topic_data = self.model.get_topic(topic_id)
                if isinstance(topic_data, list) and topic_data:
                    topic_words = [word[0] for word in topic_data]
            except Exception:
                topic_words = []
        
        ml_label = self._map_topic_to_business_label(topic_id)
        final_label = ml_label

        # 2. KEYWORD OVERRIDE
        text_lower = text.lower()
        for category, keywords in self.keyword_overrides.items():
            if any(kw in text_lower for kw in keywords):
                final_label = category
                confidence = max(confidence, 0.95) 
                break
        
        return {
            "topic_id": int(topic_id),
            "topic_label": final_label,
            "topic_words": topic_words,
            "topic_prob": confidence
        }