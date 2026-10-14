"""
Term Frequency - Inverse Document Frequency (TF-IDF) Mathematical Implementation.
"""

from collections import Counter
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


class TFIDFModel:
    """
    Mathematical TF-IDF vectorizer built from scratch with smooth IDF and L2 normalization.
    """
    def __init__(self, max_features: Optional[int] = None, smooth_idf: bool = True):
        self.max_features = max_features
        self.smooth_idf = smooth_idf
        self.vocabulary_: Dict[str, int] = {}
        self.feature_names_: List[str] = []
        self.idf_values_: np.ndarray = np.array([])

    def fit(self, documents: List[str]) -> "TFIDFModel":
        """Compute IDF weights across document collection."""
        n_docs = len(documents)
        term_doc_counts: Counter = Counter()
        all_terms: Counter = Counter()

        tokenized_docs = [doc.lower().split() for doc in documents]
        for doc in tokenized_docs:
            unique_terms = set(doc)
            term_doc_counts.update(unique_terms)
            all_terms.update(doc)

        if self.max_features:
            most_common = all_terms.most_common(self.max_features)
            self.feature_names_ = sorted([term for term, _ in most_common])
        else:
            self.feature_names_ = sorted(list(term_doc_counts.keys()))

        self.vocabulary_ = {term: idx for idx, term in enumerate(self.feature_names_)}

        # Compute smooth IDF: ln((1 + N) / (1 + DF)) + 1
        idf = np.zeros(len(self.feature_names_))
        for idx, term in enumerate(self.feature_names_):
            df = term_doc_counts.get(term, 0)
            if self.smooth_idf:
                idf[idx] = math.log((1 + n_docs) / (1 + df)) + 1.0
            else:
                idf[idx] = math.log(n_docs / (df if df > 0 else 1.0))

        self.idf_values_ = idf
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to TF-IDF normalized vectors."""
        if not self.vocabulary_:
            raise ValueError("TFIDFModel must be fitted before transforming.")

        n_docs = len(documents)
        n_features = len(self.feature_names_)
        tfidf_matrix = np.zeros((n_docs, n_features), dtype=float)

        for doc_idx, doc in enumerate(documents):
            words = doc.lower().split()
            doc_len = len(words)
            if doc_len == 0:
                continue

            tf_counts = Counter(words)
            for word, count in tf_counts.items():
                if word in self.vocabulary_:
                    term_idx = self.vocabulary_[word]
                    tf = count / doc_len
                    tfidf_matrix[doc_idx, term_idx] = tf * self.idf_values_[term_idx]

            # L2 Vector Normalization
            norm = np.linalg.norm(tfidf_matrix[doc_idx])
            if norm > 0:
                tfidf_matrix[doc_idx] /= norm

        return tfidf_matrix

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)

    def get_top_keywords(self, document: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Extract top highest-weighted TF-IDF keywords for a document."""
        vec = self.transform([document])[0]
        top_indices = np.argsort(vec)[::-1][:top_k]
        return [(self.feature_names_[idx], float(vec[idx])) for idx in top_indices if vec[idx] > 0]
