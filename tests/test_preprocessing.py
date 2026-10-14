"""
Tests for Text Cleaning, Tokenization, Morphology, and Ambiguity.
"""

import pytest

from nlp_engine.preprocessing.cleaner import TextCleaner
from nlp_engine.preprocessing.tokenizer import WordTokenizer, SentenceTokenizer, BytePairEncoder
from nlp_engine.preprocessing.morphology import MorphologyProcessor
from nlp_engine.preprocessing.ambiguity import AmbiguityHandler


def test_text_cleaner():
    cleaner = TextCleaner()
    raw = "Check http://example.com/order and email test@domain.com! Order #1234."
    cleaned = cleaner.clean(raw)
    assert "http" not in cleaned
    assert "test@domain.com" not in cleaned
    assert "!" not in cleaned
    assert "order" in cleaned


def test_word_and_sentence_tokenizers():
    word_tok = WordTokenizer()
    sent_tok = SentenceTokenizer()

    text = "Hello Dr. Smith. How can I help you today? Please advise."
    sents = sent_tok.tokenize(text)
    assert len(sents) == 3  # Dr. does not split, leaving 3 distinct sentences

    words = word_tok.tokenize(sents[0])
    assert "hello" in words


def test_byte_pair_encoder():
    bpe = BytePairEncoder(num_merges=10)
    corpus = ["lower", "lowest", "newer", "newest", "wider", "widest"]
    bpe.fit(corpus)
    tokens = bpe.tokenize("lowest")
    assert len(tokens) > 0
    assert any("est" in t or "ow" in t or "low" in t for t in tokens)


def test_morphology_processor():
    morph = MorphologyProcessor()
    tokens = ["the", "quick", "running", "orders", "are", "here"]
    filtered = morph.remove_stopwords(tokens)
    assert "the" not in filtered
    assert "are" not in filtered
    assert "running" in filtered

    stemmed = morph.stem("running")
    assert stemmed.startswith("run")

    lemmatized = morph.lemmatize("orders", pos="n")
    assert lemmatized == "order"

    comparison = morph.compare_stem_vs_lemma(["running", "orders"])
    assert len(comparison) == 2


def test_ambiguity_handler():
    ambig = AmbiguityHandler()
    res_verb = ambig.detect_syntactic_part_of_speech("book", "to", "a")
    assert res_verb == "VERB"

    res_noun = ambig.detect_syntactic_part_of_speech("book", "the", "carefully")
    assert res_noun == "NOUN"

    sense = ambig.disambiguate_word("charge", ["credit", "card", "refund"])
    assert sense["sense"] == "financial"
