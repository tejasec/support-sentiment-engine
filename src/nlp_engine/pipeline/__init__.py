"""
Pipeline orchestration and production FastAPI serving layer.
"""

from nlp_engine.pipeline.orchestrator import NLPOrchestrator
from nlp_engine.pipeline.api import app

__all__ = ["NLPOrchestrator", "app"]
