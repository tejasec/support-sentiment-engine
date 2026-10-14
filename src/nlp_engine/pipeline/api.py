"""
Production FastAPI Serving Layer for Intelligent Customer Support.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# Pydantic Schemas
class TicketRequest(BaseModel):
    text: str = Field(..., description="Customer ticket content or inquiry message")


class SentimentRequest(BaseModel):
    text: str = Field(..., description="Customer review or feedback text")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User chat message to support assistant")


class HealthResponse(BaseModel):
    status: str
    version: str
    system: str


if FASTAPI_AVAILABLE:
    from nlp_engine.pipeline.orchestrator import NLPOrchestrator
    from nlp_engine.data.generator import generate_support_tickets

    app = FastAPI(
        title="Intelligent Customer Support NLP Engine",
        description="Multi-stage NLP API for ticket triage, sentiment analytics, NER, and conversational resolution.",
        version="0.1.0"
    )

    # Initialize and pre-warm orchestrator with training data
    orchestrator = NLPOrchestrator()
    _warmup_df = generate_support_tickets(n_samples=200, random_state=42)
    orchestrator.fit(_warmup_df)

    @app.get("/health", response_model=HealthResponse)
    def health_check():
        return {
            "status": "healthy",
            "version": "0.1.0",
            "system": "Intelligent Customer Support & Sentiment Analytics Engine"
        }

    @app.post("/api/v1/predict/ticket")
    def predict_ticket(request: TicketRequest):
        try:
            return orchestrator.classifier.predict_ticket(request.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/sentiment")
    def analyze_sentiment(request: SentimentRequest):
        return {
            "polarity": orchestrator.sentiment.analyze_polarity(request.text),
            "aspects": orchestrator.sentiment.analyze_aspects(request.text)
        }

    @app.post("/api/v1/ner")
    def extract_entities(request: TicketRequest):
        return {"entities": orchestrator.ner.extract_entities(request.text)}

    @app.post("/api/v1/chat")
    def chat_response(request: ChatRequest):
        return orchestrator.chatbot.respond(request.message)

    @app.post("/api/v1/pipeline")
    def run_full_pipeline(request: TicketRequest):
        return orchestrator.process_ticket(request.text)

else:
    app = None
