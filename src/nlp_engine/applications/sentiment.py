"""
Sentiment Analysis and Opinion Mining: Polarity, Subjectivity, and Aspect-Based Sentiment.
"""

from typing import Dict, List, Optional
import re


POSITIVE_WORDS = {
    "great", "excellent", "good", "perfect", "fast", "helpful", "resolved", "love",
    "fantastic", "amazing", "thanks", "thank", "pleased", "smooth", "happy", "best"
}

NEGATIVE_WORDS = {
    "terrible", "bad", "slow", "horrible", "error", "broken", "disputed", "stuck",
    "frustrated", "delay", "damaged", "fail", "failed", "crash", "blocked", "worst",
    "awful", "unacceptable", "useless", "lost", "charge", "refund"
}

ASPECTS = {
    "billing": ["charge", "fee", "bill", "invoice", "refund", "price", "cost"],
    "delivery": ["shipping", "package", "transit", "delivery", "courier", "arrival", "box"],
    "software": ["app", "dashboard", "crash", "bug", "error", "feature", "patch", "lag"],
    "service": ["agent", "support", "team", "turnaround", "response", "representative"]
}


class SentimentAnalyzer:
    """
    Sentiment analyzer computing continuous polarity, subjectivity, and aspect sentiments.
    """
    def __init__(self, pos_threshold: float = 0.15, neg_threshold: float = -0.15):
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def analyze_polarity(self, text: str) -> Dict[str, object]:
        """
        Analyze continuous sentiment polarity and subjectivity.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return {"polarity": 0.0, "sentiment": "Neutral", "subjectivity": 0.0}

        pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
        total_eval = pos_count + neg_count

        polarity = (pos_count - neg_count) / max(1, total_eval) if total_eval > 0 else 0.0
        subjectivity = total_eval / len(words)

        if polarity >= self.pos_threshold:
            sentiment = "Positive"
        elif polarity <= self.neg_threshold:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            "polarity": round(float(polarity), 4),
            "sentiment": sentiment,
            "subjectivity": round(float(min(1.0, subjectivity * 3.0)), 4)
        }

    def analyze_aspects(self, text: str) -> Dict[str, Dict[str, object]]:
        """
        Aspect-based sentiment analysis extracting polarity specifically mapped to aspects.
        """
        sentences = re.split(r"[.!?]+", text)
        aspect_results = {}

        for aspect_name, keywords in ASPECTS.items():
            aspect_sentences = [
                s for s in sentences if any(k in s.lower() for k in keywords)
            ]
            if aspect_sentences:
                combined_text = " ".join(aspect_sentences)
                aspect_results[aspect_name] = self.analyze_polarity(combined_text)

        return aspect_results
