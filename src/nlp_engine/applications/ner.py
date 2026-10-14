"""
Named Entity Recognition (NER) for Customer Support Operations.
"""

import re
from typing import Dict, List, Optional


class CustomerNER:
    """
    Extracts structured domain entities from unstructured customer queries:
    Order IDs, Products, Dates, Contact Info, and Person names.
    """
    def __init__(self, spacy_model: Optional[str] = "en_core_web_sm"):
        self.spacy_model = spacy_model
        self.nlp = None

        # Regex patterns for high-precision support entity extraction
        self.patterns = {
            "ORDER_ID": re.compile(r"\b(?:ORD-\d+|#\d{5,8})\b", re.IGNORECASE),
            "DATE": re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b"),
            "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "PRODUCT": re.compile(
                r"\b(?:Pro Cloud Subscription|Enterprise Data Engine|Smart Home Hub v2|"
                r"Ultra Wireless Earbuds|AI Analytics Dashboard|Omni POS Terminal)\b",
                re.IGNORECASE
            )
        }

    def _init_spacy(self) -> bool:
        if self.nlp is not None:
            return True
        try:
            import spacy
            self.nlp = spacy.load(self.spacy_model)
            return True
        except Exception:
            return False

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract all detected entities with type, text span, and offsets.
        """
        entities: List[Dict[str, str]] = []

        # 1. High precision regex rules
        for label, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                entities.append({
                    "entity": match.group(),
                    "label": label,
                    "start": match.start(),
                    "end": match.end()
                })

        # 2. spaCy statistical NER augmentation (PERSON, ORG, GPE)
        if self._init_spacy():
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE"]:
                    # Avoid duplicate overlapping spans
                    if not any(e["start"] <= ent.start_char < e["end"] for e in entities):
                        entities.append({
                            "entity": ent.text,
                            "label": ent.label_,
                            "start": ent.start_char,
                            "end": ent.end_char
                        })

        return sorted(entities, key=lambda x: x["start"])
