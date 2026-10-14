"""
Word, Sentence, and Subword (BPE) Tokenizers.
Advanced Subword Tokenization.
"""

from collections import Counter, defaultdict
import re
from typing import Dict, List, Set, Tuple


class WordTokenizer:
    """Word-level tokenizer with regex boundary splitting."""
    def __init__(self, preserve_case: bool = False):
        self.preserve_case = preserve_case
        self.pattern = re.compile(r"\b\w+(?:'\w+)?\b|[^\w\s]")

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = self.pattern.findall(text)
        return tokens if self.preserve_case else [t.lower() for t in tokens]


class SentenceTokenizer:
    """Sentence boundary tokenizer with honorific/abbreviation protection."""
    ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "inc.", "ltd.", "vs.", "eg.", "ie."}

    def tokenize(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        raw_sents = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences: List[str] = []
        buffer = ""
        for s in raw_sents:
            if buffer:
                buffer += " " + s
            else:
                buffer = s
            last_word = buffer.split()[-1].lower() if buffer.split() else ""
            if last_word not in self.ABBREVIATIONS:
                sentences.append(buffer.strip())
                buffer = ""
        if buffer:
            sentences.append(buffer.strip())
        return [s for s in sentences if s]


class BytePairEncoder:
    """
    Subword Byte-Pair Encoding (BPE) Tokenizer implementation from scratch.
    """
    def __init__(self, num_merges: int = 100):
        self.num_merges = num_merges
        self.vocab: Set[str] = set()
        self.merges: List[Tuple[str, str]] = []

    def _get_stats(self, vocab: Dict[Tuple[str, ...], int]) -> Counter:
        pairs: Counter = Counter()
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        new_vocab = {}
        bigram = pair
        replacement = "".join(pair)
        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == bigram:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    def fit(self, texts: List[str]) -> "BytePairEncoder":
        """Learn BPE merge operations from input texts."""
        word_freqs: Counter = Counter()
        for text in texts:
            words = text.strip().split()
            word_freqs.update(words)

        vocab = {tuple(list(w) + ["</w>"]): freq for w, freq in word_freqs.items()}

        self.merges = []
        for _ in range(self.num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)

        self.vocab = set()
        for word in vocab:
            self.vocab.update(word)

        return self

    def tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word into subword units using learned merges."""
        tokens = list(word) + ["</w>"]
        for pair in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append("".join(pair))
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def tokenize(self, text: str) -> List[str]:
        """Tokenize full text into BPE subwords."""
        words = text.strip().split()
        subwords = []
        for w in words:
            subwords.extend(self.tokenize_word(w))
        return subwords
