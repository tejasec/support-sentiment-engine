"""
Machine Translation and Multilingual Support Evaluation (BLEU Metric).
"""

from typing import Dict, List, Optional
from nlp_engine.utils.metrics import calculate_bleu_score


# Bilingual dictionary for fast offline multilingual mapping
BASIC_TRANSLATION_DICT = {
    "es": {
        "hola": "hello",
        "gracias": "thank you",
        "ayuda": "help",
        "pedido": "order",
        "factura": "invoice",
        "cancelar": "cancel",
        "reembolso": "refund"
    },
    "fr": {
        "bonjour": "hello",
        "merci": "thank you",
        "aide": "help",
        "commande": "order",
        "facture": "invoice",
        "annuler": "cancel",
        "remboursement": "refund"
    }
}


class SupportTranslator:
    """
    Multilingual translation layer with BLEU score evaluation.
    """
    def __init__(self):
        self._hf_pipeline = None

    def translate_phrase(self, text: str, src_lang: str = "es", target_lang: str = "en") -> str:
        """Translate words/phrases using rule lookup or MarianMT."""
        words = text.lower().split()
        if src_lang in BASIC_TRANSLATION_DICT:
            mapped = [BASIC_TRANSLATION_DICT[src_lang].get(w, w) for w in words]
            return " ".join(mapped)
        return text

    def evaluate_translation(self, reference: str, hypothesis: str) -> Dict[str, object]:
        """
        Evaluate translation quality against gold standard using BLEU score.
        """
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        bleu = calculate_bleu_score(ref_tokens, hyp_tokens)
        return {
            "reference": reference,
            "hypothesis": hypothesis,
            "bleu_score": round(float(bleu), 4)
        }
