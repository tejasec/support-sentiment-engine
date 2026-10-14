#!/usr/bin/env python3
"""
CLI Model Serving Runner.
Spins up the FastAPI Intelligent Customer Support REST service via Uvicorn.
"""

import argparse
from pathlib import Path
import sys

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_engine.utils.logger import get_logger

logger = get_logger("serve")


def main():
    parser = argparse.ArgumentParser(description="Serve the NLP Engine FastAPI application.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    try:
        import uvicorn
        logger.info(f"Starting NLP Engine API server on http://{args.host}:{args.port}...")
        uvicorn.run("nlp_engine.pipeline.api:app", host=args.host, port=args.port, reload=args.reload)
    except ImportError:
        logger.error("Uvicorn is not installed. Please install it using 'uv pip install uvicorn'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
