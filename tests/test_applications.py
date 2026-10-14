"""
Tests for Full NLP Application Suite (Classification, Sentiment, NER, Bot, Translation).
"""

import pytest

from nlp_engine.applications.ticket_classifier import TicketClassifier
from nlp_engine.applications.sentiment import SentimentAnalyzer
from nlp_engine.applications.ner import CustomerNER
from nlp_engine.applications.chatbot import HybridSupportChatbot
from nlp_engine.applications.translation import SupportTranslator


def test_ticket_classifier(sample_tickets_df):
    clf = TicketClassifier()
    clf.fit(sample_tickets_df)
    res = clf.predict_ticket("URGENT: double charge on my credit card invoice ORD-9901")
    assert "category" in res
    assert "urgency" in res
    assert "is_spam" in res
    assert res["urgency"] in ["Low", "Medium", "High", "Critical"]


def test_sentiment_analyzer():
    sa = SentimentAnalyzer()
    res_pos = sa.analyze_polarity("The product fix was fantastic, great service!")
    assert res_pos["sentiment"] == "Positive"
    assert res_pos["polarity"] > 0

    res_neg = sa.analyze_polarity("Awful system crash and terrible delay.")
    assert res_neg["sentiment"] == "Negative"
    assert res_neg["polarity"] < 0

    aspects = sa.analyze_aspects("The delivery package was delayed, but customer support was great.")
    assert "delivery" in aspects


def test_customer_ner():
    ner = CustomerNER()
    text = "Please check order ORD-55421 dated 2026-05-18 for Alice Johnson at alice@example.com."
    entities = ner.extract_entities(text)
    labels = [e["label"] for e in entities]
    assert "ORDER_ID" in labels
    assert "DATE" in labels
    assert "EMAIL" in labels


def test_hybrid_chatbot():
    bot = HybridSupportChatbot()
    r1 = bot.respond("Hello there!")
    assert r1["mode"] == "rule_based"
    assert r1["intent"] == "GREETING"

    r2 = bot.respond("How do I reset my password?")
    assert r2["mode"] == "rule_based"
    assert r2["intent"] == "PASSWORD_RESET"

    r3 = bot.respond("The widget is showing error code 44")
    assert r3["mode"] in ["ml_intent", "fallback"]


def test_support_translator():
    translator = SupportTranslator()
    translated = translator.translate_phrase("hola gracias", src_lang="es")
    assert "hello" in translated
    assert "thank you" in translated

    eval_res = translator.evaluate_translation("hello world", "hello world")
    assert eval_res["bleu_score"] > 0.9
