#!/usr/bin/env python3
"""
CLI Data Generation Utility.
Synthesizes customer support ticket datasets with categories, urgencies, sentiments, and entities.
"""

import argparse
from pathlib import Path
import sys

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_engine.config import RAW_DATA_DIR
from nlp_engine.data.generator import generate_support_tickets
from nlp_engine.utils.logger import get_logger

logger = get_logger("generate_dataset")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic customer support ticket dataset.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of tickets to generate (default: 1000)")
    parser.add_argument("--output", type=str, default=str(RAW_DATA_DIR / "customer_tickets.csv"), help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--spam-ratio", type=float, default=0.08, help="Fraction of spam queries (default: 0.08)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {args.samples} synthetic support tickets (seed={args.seed})...")
    df = generate_support_tickets(n_samples=args.samples, random_state=args.seed, spam_ratio=args.spam_ratio)
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset saved to: {output_path}")

    # Print summary statistics
    print("\n--- Dataset Distribution Summary ---")
    print(f"Total Tickets : {len(df)}")
    print(f"Spam Queries  : {df['is_spam'].sum()} ({df['is_spam'].mean()*100:.1f}%)")
    print("\nCategory Breakdown:")
    for cat, count in df["category"].value_counts().items():
        print(f"  - {cat:30s}: {count:4d}")
    print("\nUrgency Breakdown:")
    for urg, count in df["urgency"].value_counts().items():
        print(f"  - {urg:15s}: {count:4d}")
    print("\nSentiment Breakdown:")
    for sent, count in df["sentiment"].value_counts().items():
        print(f"  - {sent:15s}: {count:4d}")


if __name__ == "__main__":
    main()
