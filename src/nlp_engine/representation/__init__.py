"""
Text Representation Techniques.
Provides Bag-of-Words, TF-IDF, Word2Vec/GloVe embeddings, and Contextual Transformer representations.
"""

from nlp_engine.representation.bow import BagOfWords
from nlp_engine.representation.tfidf import TFIDFModel
from nlp_engine.representation.embeddings import Word2VecEngine
from nlp_engine.representation.contextual import ContextualEmbedder

__all__ = [
    "BagOfWords",
    "TFIDFModel",
    "Word2VecEngine",
    "ContextualEmbedder",
]
