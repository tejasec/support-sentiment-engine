"""
NLP Applications & Solutions.
Provides Ticket Categorization/Spam filtering, Sentiment Analysis,
Named Entity Recognition (NER), Hybrid Chatbot, and Translation.
"""

from nlp_engine.applications.ticket_classifier import TicketClassifier
from nlp_engine.applications.sentiment import SentimentAnalyzer
from nlp_engine.applications.ner import CustomerNER
from nlp_engine.applications.chatbot import HybridSupportChatbot
from nlp_engine.applications.translation import SupportTranslator

__all__ = [
    "TicketClassifier",
    "SentimentAnalyzer",
    "CustomerNER",
    "HybridSupportChatbot",
    "SupportTranslator",
]
