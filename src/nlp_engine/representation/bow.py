"""
Bag of Words (BoW) Vectorization from Scratch and Analysis.
"""

from collections import Counter
from typing import Dict, List, Optional
import numpy as np


class BagOfWords:
    """
    Bag of Words frequency matrix builder from scratch.
    Analyzes vector sparsity and frequency distributions.
    """
    def __init__(self, max_features: Optional[int] = None, binary: bool = False):
        self.max_features = max_features
        self.binary = binary
        self.vocabulary_: Dict[str, int] = {}
        self.feature_names_: List[str] = []

    def fit(self, documents: List[str]) -> "BagOfWords":
        """Build vocabulary from document corpus."""
        term_counts: Counter = Counter()
        for doc in documents:
            words = doc.lower().split()
            term_counts.update(words)

        if self.max_features:
            most_common = term_counts.most_common(self.max_features)
            self.feature_names_ = sorted([term for term, _ in most_common])
        else:
            self.feature_names_ = sorted(list(term_counts.keys()))

        self.vocabulary_ = {term: idx for idx, term in enumerate(self.feature_names_)}
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to frequency count matrix."""
        if not self.vocabulary_:
            raise ValueError("BagOfWords must be fitted before transforming documents.")

        matrix = np.zeros((len(documents), len(self.feature_names_)), dtype=int)
        for doc_idx, doc in enumerate(documents):
            words = doc.lower().split()
            for word in words:
                if word in self.vocabulary_:
                    term_idx = self.vocabulary_[word]
                    if self.binary:
                        matrix[doc_idx, term_idx] = 1
                    else:
                        matrix[doc_idx, term_idx] += 1
        return matrix

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform in a single pass."""
        return self.fit(documents).transform(documents)

    def calculate_sparsity(self, matrix: np.ndarray) -> float:
        """
        Calculate sparsity percentage of feature matrix:
        Sparsity = (zero elements / total elements) * 100.
        """
        total_elements = matrix.size
        if total_elements == 0:
            return 0.0
        zero_elements = np.sum(matrix == 0)
        return float((zero_elements / total_elements) * 100.0)

    def get_feature_names(self) -> List[str]:
        return self.feature_names_
