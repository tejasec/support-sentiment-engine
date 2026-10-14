"""
Tests for Text Representation Techniques (BoW, TF-IDF, Word2Vec, Contextual).
"""

import numpy as np
import pytest

from nlp_engine.representation.bow import BagOfWords
from nlp_engine.representation.tfidf import TFIDFModel
from nlp_engine.representation.embeddings import Word2VecEngine
from nlp_engine.representation.contextual import ContextualEmbedder


def test_bag_of_words(sample_corpus):
    bow = BagOfWords(max_features=20)
    matrix = bow.fit_transform(sample_corpus)
    assert matrix.shape[0] == len(sample_corpus)
    assert matrix.shape[1] <= 20
    sparsity = bow.calculate_sparsity(matrix)
    assert 0.0 <= sparsity <= 100.0


def test_tfidf_model(sample_corpus):
    tfidf = TFIDFModel(max_features=25)
    matrix = tfidf.fit_transform(sample_corpus)
    assert matrix.shape[0] == len(sample_corpus)
    # Check L2 normalization: row norms should be approx 1 (or 0 if empty)
    row_norms = np.linalg.norm(matrix, axis=1)
    for norm in row_norms:
        assert np.isclose(norm, 1.0) or np.isclose(norm, 0.0)

    keywords = tfidf.get_top_keywords(sample_corpus[0], top_k=2)
    assert len(keywords) <= 2


def test_word2vec_engine(sample_corpus):
    tokenized = [doc.lower().split() for doc in sample_corpus]
    w2v = Word2VecEngine(vector_size=16, window=2, min_count=1)
    w2v.fit(tokenized)

    vec = w2v.get_word_vector("refund")
    assert vec.shape == (16,)
    doc_vec = w2v.get_document_vector(["refund", "order"])
    assert doc_vec.shape == (16,)

    sim = w2v.similarity("refund", "order")
    assert -1.0 <= sim <= 1.0


def test_contextual_embedder():
    ctx = ContextualEmbedder()
    vec = ctx.extract_sequence_embedding("Customer inquiry regarding invoice.")
    assert isinstance(vec, np.ndarray)
    assert len(vec) > 0

    poly = ctx.demonstrate_polysemy(
        "charge",
        "Disputed credit card charge.",
        "Recharge the lithium phone battery."
    )
    assert "cosine_similarity" in poly
    assert -1.0 <= poly["cosine_similarity"] <= 1.0
