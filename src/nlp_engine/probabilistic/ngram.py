"""
N-gram Language Modeling with Laplace Smoothing and Perplexity.
"""

from collections import Counter, defaultdict
import math
import random
from typing import Dict, List, Optional, Tuple


class NGramLanguageModel:
    """
    N-gram language model supporting Laplace (Add-k) smoothing.
    Calculates sequence likelihood, perplexity, and text generation.
    """
    def __init__(self, n: int = 2, alpha: float = 1.0):
        self.n = n
        self.alpha = alpha  # Smoothing parameter (alpha=1 for Laplace)
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.vocab: set = set()

    def _pad_tokens(self, tokens: List[str]) -> List[str]:
        pad = ["<s>"] * (self.n - 1)
        return pad + tokens + ["</s>"]

    def fit(self, tokenized_sentences: List[List[str]]) -> "NGramLanguageModel":
        """Train language model on tokenized sentences."""
        for sent in tokenized_sentences:
            padded = self._pad_tokens(sent)
            self.vocab.update(padded)

            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i + self.n])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

        return self

    def conditional_prob(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Calculate smoothed conditional probability P(word | context)
        using Laplace smoothing: (count(context, word) + alpha) / (count(context) + alpha * |V|).
        """
        ngram = context + (word,)
        count_ngram = self.ngram_counts[ngram]
        count_context = self.context_counts[context]
        vocab_size = len(self.vocab)

        numerator = count_ngram + self.alpha
        denominator = count_context + self.alpha * vocab_size
        return numerator / denominator if denominator > 0 else 1.0 / max(1, vocab_size)

    def calculate_sentence_perplexity(self, tokens: List[str]) -> float:
        """
        Compute Perplexity: exp(-1/N * sum(log P(w_i | context))).
        Lower perplexity indicates higher model predictability.
        """
        padded = self._pad_tokens(tokens)
        log_prob_sum = 0.0
        m = 0

        for i in range(len(padded) - self.n + 1):
            ngram = tuple(padded[i:i + self.n])
            context = ngram[:-1]
            word = ngram[-1]
            prob = self.conditional_prob(word, context)
            log_prob_sum += math.log(max(prob, 1e-12))
            m += 1

        if m == 0:
            return float("inf")

        cross_entropy = -log_prob_sum / m
        return float(math.exp(cross_entropy))

    def generate(self, prompt_tokens: Optional[List[str]] = None, max_length: int = 15) -> List[str]:
        """Greedy / sampling sequence generation from prompt."""
        if prompt_tokens:
            curr = list(prompt_tokens)
        else:
            curr = ["<s>"] * (self.n - 1)

        generated = []
        for _ in range(max_length):
            context = tuple(curr[-(self.n - 1):]) if self.n > 1 else ()
            candidates = list(self.vocab - {"<s>"})
            if not candidates:
                break
            probs = [self.conditional_prob(w, context) for w in candidates]
            total = sum(probs)
            norm_probs = [p / total for p in probs]
            next_word = random.choices(candidates, weights=norm_probs, k=1)[0]
            if next_word == "</s>":
                break
            generated.append(next_word)
            curr.append(next_word)

        return generated
