"""
Support Ticket Intent Routing, Urgency Scoring, and Spam Detection.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from nlp_engine.preprocessing.cleaner import TextCleaner
from nlp_engine.representation.tfidf import TFIDFModel
from nlp_engine.probabilistic.classifier import TicketNaiveBayes


class TicketClassifier:
    """
    Unified classifier for ticket routing, urgency assessment, and spam filtering.
    """
    def __init__(self):
        self.cleaner = TextCleaner()
        self.tfidf_category = TFIDFModel(max_features=2500)
        self.nb_category = TicketNaiveBayes(alpha=1.0)

        self.tfidf_spam = TFIDFModel(max_features=1000)
        self.nb_spam = TicketNaiveBayes(alpha=1.0)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "TicketClassifier":
        """Train models for category routing and spam detection."""
        texts = self.cleaner.clean_batch(df["text"].tolist())
        categories = df["category"].values
        spam_labels = df["is_spam"].values

        # 1. Fit Category Classifier
        X_cat = self.tfidf_category.fit_transform(texts)
        self.nb_category.fit(X_cat, categories)

        # 2. Fit Spam Classifier
        X_spam = self.tfidf_spam.fit_transform(texts)
        self.nb_spam.fit(X_spam, spam_labels)

        self.is_fitted = True
        return self

    def predict_ticket(self, text: str) -> Dict[str, Any]:
        """Triage an incoming customer ticket."""
        if not self.is_fitted:
            raise ValueError("TicketClassifier must be fitted before prediction.")

        cleaned = self.cleaner.clean(text)

        # 1. Spam check
        X_sp = self.tfidf_spam.transform([cleaned])
        spam_pred = int(self.nb_spam.predict(X_sp)[0])
        spam_prob = float(self.nb_spam.predict_proba(X_sp)[0][1]) if len(self.nb_spam.classes_) > 1 else 0.0

        # 2. Category prediction
        X_cat = self.tfidf_category.transform([cleaned])
        category_pred = str(self.nb_category.predict(X_cat)[0])
        probs = self.nb_category.predict_proba(X_cat)[0]
        confidence = float(np.max(probs))

        # 3. Urgency heuristic
        urgency = "Low"
        critical_keywords = ["urgent", "error 503", "crash", "blocked", "immediately", "disputed", "stuck"]
        high_keywords = ["refund", "cancel", "locked", "lost", "delay", "damaged"]

        text_lower = text.lower()
        if any(k in text_lower for k in critical_keywords):
            urgency = "Critical"
        elif any(k in text_lower for k in high_keywords):
            urgency = "High"
        elif "help" in text_lower or "inquiry" in text_lower:
            urgency = "Medium"

        return {
            "category": category_pred,
            "category_confidence": round(confidence, 4),
            "is_spam": bool(spam_pred == 1),
            "spam_score": round(spam_prob, 4),
            "urgency": urgency
        }
