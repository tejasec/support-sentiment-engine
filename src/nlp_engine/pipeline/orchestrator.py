"""
End-to-End NLP Lifecycle Orchestrator.
Unifies all 5 pipeline phases into a coherent processing pipeline.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from nlp_engine.preprocessing.cleaner import TextCleaner
from nlp_engine.preprocessing.tokenizer import WordTokenizer
from nlp_engine.preprocessing.morphology import MorphologyProcessor
from nlp_engine.preprocessing.ambiguity import AmbiguityHandler
from nlp_engine.representation.tfidf import TFIDFModel
from nlp_engine.applications.ticket_classifier import TicketClassifier
from nlp_engine.applications.sentiment import SentimentAnalyzer
from nlp_engine.applications.ner import CustomerNER
from nlp_engine.applications.chatbot import HybridSupportChatbot
from nlp_engine.utils.logger import get_logger

logger = get_logger("NLPOrchestrator")


class NLPOrchestrator:
    """
    Central orchestration engine processing raw tickets through linguistic analysis,
    vectorization, classification, entity extraction, and conversational response.
    """
    def __init__(self):
        self.cleaner = TextCleaner()
        self.tokenizer = WordTokenizer()
        self.morphology = MorphologyProcessor()
        self.ambiguity = AmbiguityHandler()
        self.classifier = TicketClassifier()
        self.sentiment = SentimentAnalyzer()
        self.ner = CustomerNER()
        self.chatbot = HybridSupportChatbot()
        self.is_trained = False

    def fit(self, df: pd.DataFrame) -> "NLPOrchestrator":
        """Fit all train-dependent modules (TF-IDF, Naive Bayes, Classifiers)."""
        logger.info(f"Training NLP Orchestrator on {len(df)} customer ticket samples...")
        self.classifier.fit(df)
        self.is_trained = True
        logger.info("NLP Orchestrator training complete.")
        return self

    def process_ticket(self, raw_text: str) -> Dict[str, Any]:
        """
        Execute unified 5-stage NLP pipeline on a raw user ticket.
        """
        # Stage 1: Linguistic Normalization & Preprocessing
        cleaned_text = self.cleaner.clean(raw_text)
        tokens = self.tokenizer.tokenize(cleaned_text)
        filtered_tokens = self.morphology.remove_stopwords(tokens)
        stemmed_tokens = [self.morphology.stem(t) for t in filtered_tokens]

        # Stage 2 & 3: Triage & Classification
        triage_result = self.classifier.predict_ticket(raw_text) if self.is_trained else {
            "category": "Pending", "urgency": "Medium", "is_spam": False, "spam_score": 0.0
        }

        # Stage 4: Sentiment & Opinion Mining
        sentiment_result = self.sentiment.analyze_polarity(raw_text)
        aspects_result = self.sentiment.analyze_aspects(raw_text)

        # Stage 5: Information Extraction & Entities
        entities = self.ner.extract_entities(raw_text)

        # Stage 6: Conversational Resolution
        chat_reply = self.chatbot.respond(raw_text)

        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "token_count": len(tokens),
            "tokens_stemmed": stemmed_tokens[:10],
            "triage": triage_result,
            "sentiment": sentiment_result,
            "aspects": aspects_result,
            "entities": entities,
            "bot_resolution": chat_reply
        }
