"""
Linguistic Ambiguity Handling (Lexical, Syntactic, Semantic).
"""

from typing import Dict, List, Optional, Tuple


class AmbiguityHandler:
    """
    Analyzes and resolves syntactic and lexical ambiguities in customer queries.
    """
    def __init__(self):
        # Known polysemous customer keywords with domain senses
        self.lexical_rules = {
            "charge": {
                "financial": ["credit", "card", "fee", "bill", "invoice", "refund", "dollar", "$", "cost"],
                "technical": ["battery", "phone", "device", "power", "plug", "cable", "watt"]
            },
            "book": {
                "verb": ["flight", "appointment", "slot", "ticket", "demo", "hotel", "seat", "a", "to"],
                "noun": ["read", "page", "manual", "guide", "documentation", "author", "cover"]
            },
            "order": {
                "verb": ["buy", "purchase", "place", "cancel", "want", "to", "would", "like"],
                "noun": ["number", "id", "status", "tracking", "history", "details", "receipt", "my"]
            }
        }

    def disambiguate_word(self, target_word: str, context_tokens: List[str]) -> Dict[str, object]:
        """
        Disambiguate word sense based on co-occurring context words.
        """
        word = target_word.lower()
        if word not in self.lexical_rules:
            return {"word": target_word, "sense": "unambiguous", "confidence": 1.0}

        senses = self.lexical_rules[word]
        context_lower = [t.lower() for t in context_tokens if t.lower() != word]

        scores: Dict[str, int] = {sense: 0 for sense in senses}
        for token in context_lower:
            for sense, cues in senses.items():
                if token in cues:
                    scores[sense] += 1

        best_sense = max(scores, key=scores.get)
        max_score = scores[best_sense]
        total_score = sum(scores.values())

        confidence = (max_score / total_score) if total_score > 0 else 0.5
        return {
            "word": target_word,
            "sense": best_sense if max_score > 0 else "neutral",
            "scores": scores,
            "confidence": confidence
        }

    def detect_syntactic_part_of_speech(self, target_word: str, prev_word: Optional[str], next_word: Optional[str]) -> str:
        """
        Heuristic POS tag disambiguation based on immediate local syntactic context.
        E.g., "to book" -> VERB, "a book" -> NOUN.
        """
        target = target_word.lower()
        prev_w = (prev_word or "").lower()
        next_w = (next_word or "").lower()

        if prev_w in ["to", "please", "can", "will", "i", "we"]:
            return "VERB"
        if prev_w in ["a", "an", "the", "this", "my", "your", "his", "her"]:
            return "NOUN"
        if next_w in ["a", "the", "my", "an"] and target in ["book", "order", "charge"]:
            return "VERB"

        return "UNKNOWN"
