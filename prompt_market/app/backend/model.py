import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Konfigurace loggeru
logger = logging.getLogger(__name__)

class TopicModelClassifier:
    """
    Wrapper třída pro BERTopic model poskytující hybridní klasifikaci.

    Kombinuje nesupervizované učení (BERTopic) s pravidlovým systémem
    klíčových slov pro zvýšení přesnosti v doméně HR a IT.

    Attributes:
        model_path (Path): Cesta k souboru modelu (.pkl).
        default_label (str): Výchozí štítek pro nezařazená témata.
        keyword_overrides (dict): Slovník pravidel pro prioritní klasifikaci.
    """

    def __init__(self):
        """
        Inicializuje konfiguraci klasifikátoru.
        Model se nenačítá automaticky v __init__, ale až voláním load_model().
        """
        self.model = None
        # POUŽITÍ PATHLIB: Absolutní cesta relativně k tomuto souboru
        self.model_path = Path(__file__).parent / "bertopic_model.pkl"
        
        self.default_label = "General Work Task"

        # Hybridní logika: Pravidla pro specifické byznys kategorie
        self.keyword_overrides = {
            "Technical & Coding": [
                "python", "javascript", "code", "script", "function", "java ", "c++", 
                "sql", "api", "bug", "terminal", "bash", "git", "regex", "json",
                "docker", "kubernetes", "deploy", "stack trace", "debug"
            ],
            "HR & Employment Policies": [
                "holiday", "vacation", "leave", "sick day", "benefits", "insurance",
                "promotion", "hiring", "onboarding", "interview", "salary", "bonus",
                "contract", "resignation"
            ],
            "Employee Wellbeing (Flagged)": [
                "burnout", "stress", "tired", "overwhelmed", "anxiety", "mental health",
                "toxic", "harassment", "unhappy", "quit", "depression"
            ],
            "IT Support & Infrastructure": [
                "wifi", "vpn", "password", "login", "cant access", "printer", "laptop",
                "connection", "server down", "2fa", "authentication"
            ],
            "Legal & Compliance": [
                "nda", "gdpr", "policy", "regulation", "law", "legal",
                "compliance", "audit", "agreement", "privacy"
            ],
            "Finance & Budgeting": [
                "invoice", "budget", "cost", "expense", "tax", "vat", 
                "billing", "payment", "receipt", "po number"
            ],
            "Learning & Development": [
                "how to", "tutorial", "explain", "learn", "course", "training", 
                "certification", "guide", "documentation"
            ],
            "Communication & Translation": [
                "translate", "english", "german", "french", "spanish", "proofread",
                "grammar", "rewrite", "email", "summary", "tone"
            ]
        }
        
        self.business_categories = {
             0: "General Assistance",
             1: "Technical Support",
             2: "Content Generation"
        }

    def load_model(self) -> None:
        """
        Načte artefakt BERTopic modelu a embedding model z disku.
        
        Používá pathlib pro kontrolu existence souboru.
        """
        try:
            if self.model_path.exists():
                logger.info(f"Načítám model z: {self.model_path}")
                # Načtení embedding modelu pro konzistenci
                embedding_model = SentenceTransformer("all-mpnet-base-v2") 
                self.model = BERTopic.load(str(self.model_path), embedding_model=embedding_model)
                logger.info("Model úspěšně načten.")
            else:
                logger.error(f"Soubor modelu nenalezen na cestě: {self.model_path}")
        except Exception as e:
            logger.error(f"Kritická chyba při načítání modelu: {e}")

    def _map_topic_to_business_label(self, topic_id: int) -> str:
        """
        Převádí ID tématu z BERTopic na čitelnou byznys kategorii.

        Args:
            topic_id (int): Numerické ID tématu.

        Returns:
            str: Název kategorie.
        """
        if topic_id == -1:
            return "Uncategorized / Noise"
        
        if not self.model:
            return self.default_label
        
        # 1. Manuální mapping
        if topic_id in self.business_categories:
            return self.business_categories[topic_id]

        # 2. Získání informací z modelu
        try:
            info = self.model.get_topic_info(topic_id)
            if info.empty:
                return self.default_label
            topic_name = info['Name'].values[0]
            clean_name = topic_name.lower()
        except Exception:
            return f"Topic {topic_id}"
        
        # 3. Pokus o nalezení klíčových slov v názvu tématu
        for category, keywords in self.keyword_overrides.items():
            if any(k in clean_name for k in keywords):
                return category

        # 4. Fallback: Vrátí název generovaný modelem
        parts = clean_name.split('_')
        if len(parts) > 1:
            return f"Topic: {', '.join(parts[1:4])}"
        
        return f"Topic {topic_id}"

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predikuje téma pro zadaný text.

        Postup:
        1. Standardní predikce pomocí BERTopic modelu.
        2. "Keyword Override" - pokud text obsahuje specifická klíčová slova,
           přebije výsledek modelu (zajišťuje přesnost pro kritická témata).

        Args:
            text (str): Vstupní text k analýze.

        Returns:
            Dict[str, Any]: Slovník s klíči topic_id, topic_label, topic_words, topic_prob.
        """
        if not self.model:
            # Pokus o záchranu, pokud někdo zavolal predict před load_model
            self.load_model()
            if not self.model:
                raise ValueError("Model není inicializován. Zavolejte load_model().")
        
        if not text or not text.strip():
             return {
                "topic_id": -1,
                "topic_label": "Uncategorized / Noise",
                "topic_words": [],
                "topic_prob": 0.0
             }

        # --- KROK 1: Predikce modelem ---
        topics, probs = self.model.transform([text])
        topic_id = topics[0]
        
        confidence = 0.0
        if probs is not None:
            if isinstance(probs, np.ndarray) and probs.size > 0:
                confidence = float(np.max(probs))

        topic_words = []
        if topic_id != -1 and self.model:
            try:
                topic_data = self.model.get_topic(topic_id)
                if isinstance(topic_data, (list, tuple)):
                    topic_words = [word[0] for word in topic_data if isinstance(word, (list, tuple)) and len(word) > 0]
            except Exception:
                pass
        
        final_label = self._map_topic_to_business_label(topic_id)

        # --- KROK 2: Hybridní Override (Pravidla) ---
        text_lower = text.lower()
        for category, keywords in self.keyword_overrides.items():
            if any(kw in text_lower for kw in keywords):
                final_label = category
                confidence = max(confidence, 0.95) # Vysoká jistota pro pravidlovou shodu
                break
        
        return {
            "topic_id": int(topic_id),
            "topic_label": final_label,
            "topic_words": topic_words,
            "topic_prob": confidence
        }