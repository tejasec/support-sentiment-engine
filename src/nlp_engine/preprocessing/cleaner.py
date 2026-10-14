"""
Text cleaning and normalization pipeline for raw customer support queries.
"""

import re
import string
import unicodedata
from typing import List, Optional


class TextCleaner:
    """
    Modular text cleaner for normalizing noisy user inputs.
    """
    def __init__(
        self,
        lowercase: bool = True,
        strip_punctuation: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_html: bool = True,
        strip_numbers: bool = False
    ):
        self.lowercase = lowercase
        self.strip_punctuation = strip_punctuation
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_html = remove_html
        self.strip_numbers = strip_numbers

        # Precompiled regex patterns
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        self.html_pattern = re.compile(r"<.*?>")
        self.whitespace_pattern = re.compile(r"\s+")
        self.punctuation_pattern = re.compile(f"[{re.escape(string.punctuation)}]")

    def clean(self, text: Optional[str]) -> str:
        """Execute full cleaning and normalization on single text."""
        if not text or not isinstance(text, str):
            return ""

        # Normalize unicode (decompose accented characters)
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")

        if self.remove_html:
            text = self.html_pattern.sub(" ", text)

        if self.remove_urls:
            text = self.url_pattern.sub(" ", text)

        if self.remove_emails:
            text = self.email_pattern.sub(" ", text)

        if self.lowercase:
            text = text.lower()

        if self.strip_punctuation:
            text = self.punctuation_pattern.sub(" ", text)

        if self.strip_numbers:
            text = re.sub(r"\d+", " ", text)

        # Normalize whitespace
        text = self.whitespace_pattern.sub(" ", text).strip()
        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Apply cleaning across a list of texts."""
        return [self.clean(t) for t in texts]
