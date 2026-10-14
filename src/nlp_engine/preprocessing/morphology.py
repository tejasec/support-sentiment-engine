"""
Morphological Processing: Stemming vs. Lemmatization and Stopword Removal.
"""

from typing import Dict, List, Optional, Set, Tuple


# Standard English Stopwords list for self-contained, offline operation
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


class MorphologyProcessor:
    """
    Morphological normalization engine supporting Porter Stemmer and WordNet Lemmatizer.
    """
    def __init__(self, stopwords: Optional[Set[str]] = None):
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        self._porter = None
        self._lemmatizer = None

    def _get_porter(self):
        if self._porter is None:
            try:
                from nltk.stem import PorterStemmer
                self._porter = PorterStemmer()
            except ImportError:
                class SimplePorter:
                    def stem(self, w: str) -> str:
                        if w.endswith("ing") and len(w) > 5:
                            return w[:-3]
                        if w.endswith("ed") and len(w) > 4:
                            return w[:-2]
                        if w.endswith("s") and len(w) > 3:
                            return w[:-1]
                        return w
                self._porter = SimplePorter()
        return self._porter

    def _get_lemmatizer(self):
        if self._lemmatizer is None:
            try:
                from nltk.stem import WordNetLemmatizer
                self._lemmatizer = WordNetLemmatizer()
            except ImportError:
                class SimpleLemma:
                    def lemmatize(self, w: str, pos: str = "n") -> str:
                        if w.endswith("ies") and len(w) > 4:
                            return w[:-3] + "y"
                        if w.endswith("es") and len(w) > 4:
                            return w[:-2]
                        if w.endswith("s") and len(w) > 3:
                            return w[:-1]
                        return w
                self._lemmatizer = SimpleLemma()
        return self._lemmatizer

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Filter out stopwords from token list."""
        return [t for t in tokens if t.lower() not in self.stopwords]

    def stem(self, word: str) -> str:
        """Apply Porter stemming to reduce word to inflectional root."""
        return self._get_porter().stem(word.lower())

    def lemmatize(self, word: str, pos: str = "n") -> str:
        """
        Apply WordNet lemmatization with POS tag.
        POS can be 'n' (noun), 'v' (verb), 'a' (adj), 'r' (adv).
        """
        try:
            return self._get_lemmatizer().lemmatize(word.lower(), pos=pos)
        except Exception:
            # Fallback if NLTK wordnet data package not downloaded yet
            return word.lower()

    def compare_stem_vs_lemma(self, words: List[str]) -> List[Dict[str, str]]:
        """
        Systematic comparison between Porter Stemming and WordNet Lemmatization.
        """
        results = []
        for w in words:
            results.append({
                "word": w,
                "stemmed": self.stem(w),
                "lemmatized_noun": self.lemmatize(w, pos="n"),
                "lemmatized_verb": self.lemmatize(w, pos="v")
            })
        return results
