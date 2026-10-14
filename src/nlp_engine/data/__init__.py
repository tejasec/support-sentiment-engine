"""
Data synthesis, ingestion, and dataset adapters.
"""

from nlp_engine.data.generator import generate_support_tickets
from nlp_engine.data.dataset import TicketDataset, load_ticket_dataframe

__all__ = ["generate_support_tickets", "TicketDataset", "load_ticket_dataframe"]
