"""
Distributed Semantic Representations: Word2Vec (CBOW & Skip-Gram) and GloVe Vectors.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class Word2VecEngine:
    """
    Word2Vec trainer and embedding retrieval engine.
    Supports CBOW (sg=0) and Skip-Gram (sg=1).
    """
    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        sg: int = 1,
        epochs: int = 10
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.model = None
        self.vectors_: Dict[str, np.ndarray] = {}

    def fit(self, tokenized_sentences: List[List[str]]) -> "Word2VecEngine":
        """Train Word2Vec embeddings on tokenized corpus."""
        try:
            from gensim.models import Word2Vec
            self.model = Word2Vec(
                sentences=tokenized_sentences,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                sg=self.sg,
                epochs=self.epochs
            )
            self.vectors_ = {word: self.model.wv[word] for word in self.model.wv.index_to_key}
        except ImportError:
            # Fallback: Co-occurrence based SVD representation
            self._fit_svd_fallback(tokenized_sentences)
        return self

    def _fit_svd_fallback(self, tokenized_sentences: List[List[str]]) -> None:
        """Lightweight SVD fallback when gensim is unavailable."""
        vocab = sorted(list({w for sent in tokenized_sentences for w in sent}))
        word_to_id = {w: i for i, w in enumerate(vocab)}
        V = len(vocab)
        co_mat = np.zeros((V, V))

        for sent in tokenized_sentences:
            for i, target in enumerate(sent):
                target_id = word_to_id[target]
                start = max(0, i - self.window)
                end = min(len(sent), i + self.window + 1)
                for j in range(start, end):
                    if i != j:
                        ctx_id = word_to_id[sent[j]]
                        co_mat[target_id, ctx_id] += 1

        k = min(self.vector_size, V)
        U, S, _ = np.linalg.svd(co_mat)
        emb = U[:, :k] * np.sqrt(S[:k])
        self.vectors_ = {w: emb[word_to_id[w]] for w in vocab}

    def get_word_vector(self, word: str) -> np.ndarray:
        """Get vector for word or zero vector if out of vocabulary."""
        return self.vectors_.get(word.lower(), np.zeros(self.vector_size))

    def get_document_vector(self, tokens: List[str]) -> np.ndarray:
        """Mean pooling across token embeddings for document representation."""
        vecs = [self.get_word_vector(t) for t in tokens if t.lower() in self.vectors_]
        if not vecs:
            return np.zeros(self.vector_size)
        return np.mean(vecs, axis=0)

    def similarity(self, word1: str, word2: str) -> float:
        """Calculate cosine similarity between two words."""
        v1 = self.get_word_vector(word1)
        v2 = self.get_word_vector(word2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))


def load_glove_vectors(file_path: str, max_words: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Parse pre-trained GloVe vector text file into dictionary."""
    embeddings: Dict[str, np.ndarray] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_words and idx >= max_words:
                break
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings
