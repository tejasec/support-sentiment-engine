"""
Multinomial Naive Bayes Text Classifier for Support Ticket Routing.
"""

from typing import Dict, List, Optional
import numpy as np


class TicketNaiveBayes:
    """
    Multinomial Naive Bayes classifier implementation with Laplace smoothing.
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_: np.ndarray = np.array([])
        self.class_log_prior_: np.ndarray = np.array([])
        self.feature_log_prob_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TicketNaiveBayes":
        """
        Fit Naive Bayes on word count / TF-IDF matrix X and class labels y.
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            # Class prior: count(c) / total_samples
            count_cls = len(X_cls)
            self.class_log_prior_[idx] = np.log(count_cls / n_samples)

            # Feature likelihood with Laplace smoothing
            feature_counts = np.sum(X_cls, axis=0) + self.alpha
            total_feature_counts = np.sum(feature_counts)
            self.feature_log_prob_[idx] = np.log(feature_counts / total_feature_counts)

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculate unnormalized log posterior for each class."""
        # shape: (n_samples, n_classes) = X @ feature_log_prob.T + class_log_prior
        return np.dot(X, self.feature_log_prob_.T) + self.class_log_prior_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class with maximum log posterior probability."""
        log_probs = self.predict_log_proba(X)
        best_indices = np.argmax(log_probs, axis=1)
        return self.classes_[best_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute normalized posterior probabilities using softmax."""
        log_probs = self.predict_log_proba(X)
        # Shift for numerical stability
        max_log = np.max(log_probs, axis=1, keepdims=True)
        exp_probs = np.exp(log_probs - max_log)
        return exp_probs / np.sum(exp_probs, axis=1, keepdims=True)
