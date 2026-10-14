.PHONY: help init-env install data phase-1 phase-2 phase-3 phase-4 phase-5 pipeline test serve clean

help:
	@echo "NLP Engine - Automation"
	@echo "--------------------------------------------------"
	@echo "init-env   : Create virtual environment using Python 3.12 via uv"
	@echo "install    : Install editable package with dev dependencies"
	@echo "data       : Generate synthetic customer support dataset"
	@echo "phase-1    : Run Phase 1: Preprocessing & Morphology"
	@echo "phase-2    : Run Phase 2: Text Representation & Embeddings"
	@echo "phase-3    : Run Phase 3: Probabilistic & Sequence Models"
	@echo "phase-4    : Run Phase 4: Deep Learning & Transformers"
	@echo "phase-5    : Run Phase 5: Full Applications Suite"
	@echo "pipeline   : Execute complete end-to-end processing pipeline"
	@echo "test       : Run full automated pytest suite"
	@echo "serve      : Start FastAPI model serving backend"
	@echo "clean      : Remove temporary files, build caches, and test artifacts"

init-env:
	uv venv --python 3.12 .venv

install:
	uv pip install -e ".[dev]"

data:
	python scripts/generate_dataset.py --samples 1000 --output data/raw/customer_tickets.csv

phase-1:
	python scripts/run_phase.py --phase 1

phase-2:
	python scripts/run_phase.py --phase 2

phase-3:
	python scripts/run_phase.py --phase 3

phase-4:
	python scripts/run_phase.py --phase 4

phase-5:
	python scripts/run_phase.py --phase 5

pipeline:
	python scripts/run_pipeline.py

test:
	pytest tests/ -v

serve:
	python scripts/serve.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov
