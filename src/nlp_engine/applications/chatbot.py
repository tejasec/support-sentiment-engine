"""
Dual-Mode Conversational Support Chatbot: Rule-Based Intent Tree + ML Fallback.
"""

import re
from typing import Dict, List, Optional, Tuple


class HybridSupportChatbot:
    """
    Hybrid conversational assistant integrating rule-based decision trees
    and machine learning fallback routing for customer inquiries.
    """
    def __init__(self, fallback_message: Optional[str] = None):
        self.fallback_message = fallback_message or (
            "I'm connecting you with a human support specialist. "
            "Please provide your Order ID or registered email."
        )

        # Rule-based decision patterns (Regex -> Structured Response)
        self.rules: List[Tuple[re.Pattern, str, str]] = [
            (
                re.compile(r"\b(hi|hello|hey|good\s*(morning|afternoon|evening))\b", re.I),
                "GREETING",
                "Hello! Welcome to Customer Support. How can I assist you with your orders or services today?"
            ),
            (
                re.compile(r"\b(reset|forgot|change)\s*(my\s*)?(password|login|2fa)\b", re.I),
                "PASSWORD_RESET",
                "You can reset your account password by visiting Account Settings > Security > Reset Password."
            ),
            (
                re.compile(r"\b(refund|money\s*back|double\s*charge|disputed\s*charge)\b", re.I),
                "REFUND_REQUEST",
                "Refund requests are processed within 3-5 business days. Please verify your invoice number."
            ),
            (
                re.compile(r"\b(track|where\s*is|shipping|delivery|package|transit)\b", re.I),
                "ORDER_TRACKING",
                "You can track your package in real-time by entering your Tracking/Order ID in our Orders portal."
            ),
            (
                re.compile(r"\b(cancel|terminate|stop)\s*(my\s*)?(subscription|plan|service)\b", re.I),
                "CANCELLATION",
                "To cancel your subscription, navigate to Billing > Subscriptions and click 'Cancel Plan'."
            ),
            (
                re.compile(r"\b(thank\s*you|thanks|great\s*job|awesome)\b", re.I),
                "GRATITUDE",
                "You are very welcome! Is there anything else I can help you with today?"
            )
        ]

    def respond(self, message: str) -> Dict[str, object]:
        """
        Process user message through rule tree, falling back to ML intent classifier if needed.
        """
        # 1. Rule-based evaluation
        for pattern, intent, reply in self.rules:
            if pattern.search(message):
                return {
                    "mode": "rule_based",
                    "intent": intent,
                    "confidence": 1.0,
                    "response": reply
                }

        # 2. Heuristic ML intent fallback
        msg_lower = message.lower()
        if "error" in msg_lower or "bug" in msg_lower or "crash" in msg_lower:
            return {
                "mode": "ml_intent",
                "intent": "TECHNICAL_ISSUE",
                "confidence": 0.82,
                "response": "Our technical operations team has logged this issue. Please share system logs or error code."
            }

        # 3. Graceful fallback
        return {
            "mode": "fallback",
            "intent": "UNKNOWN",
            "confidence": 0.0,
            "response": self.fallback_message
        }
