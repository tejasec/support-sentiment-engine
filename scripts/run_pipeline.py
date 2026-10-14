#!/usr/bin/env python3
"""
Full End-to-End NLP Lifecycle Runner.
Orchestrates ingestion, preprocessing, vectorization, modeling, evaluation, and application demo.
"""

import json
from pathlib import Path
import sys

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_engine.config import METRICS_DIR, RAW_DATA_DIR
from nlp_engine.data.generator import generate_support_tickets
from nlp_engine.data.dataset import split_tickets
from nlp_engine.pipeline.orchestrator import NLPOrchestrator
from nlp_engine.utils.logger import get_logger
from nlp_engine.utils.metrics import calculate_classification_metrics

logger = get_logger("run_pipeline")


def main():
    logger.info("Starting End-to-End NLP Production Pipeline...")

    # Step 1: Ingestion / Generation
    data_file = RAW_DATA_DIR / "customer_tickets.csv"
    if not data_file.exists():
        logger.info(f"Generating synthetic tickets at {data_file}...")
        df = generate_support_tickets(n_samples=500, random_state=42)
        df.to_csv(data_file, index=False)
    else:
        logger.info(f"Loading existing tickets from {data_file}...")
        import pandas as pd
        df = pd.read_csv(data_file)

    # Step 2: Data Split
    train_df, val_df, test_df = split_tickets(df, train_size=0.7, val_size=0.15, test_size=0.15)
    logger.info(f"Splits -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Step 3: Pipeline Training
    orchestrator = NLPOrchestrator()
    orchestrator.fit(train_df)

    # Step 4: Model Evaluation on Test Split
    test_texts = test_df["text"].tolist()
    y_true = test_df["category"].tolist()
    y_pred = [orchestrator.classifier.predict_ticket(t)["category"] for t in test_texts]

    metrics = calculate_classification_metrics(y_true, y_pred)
    logger.info(f"Test Accuracy       : {metrics['accuracy']*100:.2f}%")
    logger.info(f"Test Macro F1-Score : {metrics['macro_f1']:.4f}")

    # Step 5: Save Metrics Artifact
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / "pipeline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved pipeline metrics to: {metrics_path}")

    # Step 6: Sample Inference Execution
    sample_ticket = (
        "Hello Support, my order ORD-88192 for the Enterprise Data Engine "
        "has been delayed since 2026-03-15. Where is my delivery? Need urgent update!"
    )
    result = orchestrator.process_ticket(sample_ticket)

    print("\n" + "=" * 80)
    print("PIPELINE SAMPLE INFERENCE RESULT")
    print("=" * 80)
    print(f"Input Ticket      : {sample_ticket}")
    print(f"Cleaned Text      : {result['cleaned_text']}")
    print(f"Predicted Category: {result['triage']['category']} (Confidence: {result['triage']['category_confidence']})")
    print(f"Urgency Level     : {result['triage']['urgency']}")
    print(f"Overall Sentiment : {result['sentiment']['sentiment']} (Polarity: {result['sentiment']['polarity']})")
    print(f"Extracted Entities: {result['entities']}")
    print(f"Chatbot Action    : [{result['bot_resolution']['mode']}] {result['bot_resolution']['response']}")
    print("=" * 80 + "\n")
    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
