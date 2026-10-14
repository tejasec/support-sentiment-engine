# End-to-End Intelligent Customer Support & Sentiment Analytics System

A modular, production-ready Natural Language Processing pipeline and microservice suite. This project progresses from raw text ingestion and morphological normalization through distributed embeddings, probabilistic language models, deep sequence architectures, and full real-world applications (intent routing, sentiment analytics, named entity extraction, and conversational bots).

---

## 🎯 System Architecture Overview

This project implements a multi-phase NLP processing pipeline:

```
Raw Customer Support Tickets & Queries
       │
       ▼
Phase 1: Linguistic Foundations ──► Modular Cleaning, Subword Tokenizers, Stemming vs. Lemmatization
       │
       ▼
Phase 2: Text Vectorization     ──► BoW, TF-IDF, Word2Vec (CBOW/Skip-gram), GloVe, BERT Embeddings
       │
       ▼
Phase 3: Probabilistic Models   ──► N-grams (Laplace Smoothing), HMM POS Tagging, Naive Bayes Routing
       │
       ▼
Phase 4: Deep Learning in NLP   ──► PyTorch MLP, Vanilla RNN (Gradient vanishing), LSTM, GRU, Transformers
       │
       ▼
Phase 5: Applications & API     ──► Ticket Classifier, Sentiment Engine, NER, Hybrid Chatbot, FastAPI
```

---

## 📂 Repository Structure

```
NLP/
├── configs/
│   ├── config.yaml                    # Global paths, seeds, and dataset parameters
│   └── model_configs.yaml             # Hyperparameters for all models and tokenizers
├── data/
│   ├── raw/                           # Raw synthetic / ingested customer tickets
│   ├── processed/                     # Tokenized, vectorized, train/val/test splits
│   └── embeddings/                    # Word2Vec and pre-trained embedding vectors
├── artifacts/
│   ├── models/                        # Serialized classifiers, PyTorch weights, vocabularies
│   ├── metrics/                       # Evaluation reports, confusion matrices, test logs
│   ├── plots/                         # t-SNE projections, loss curves, confusion matrices
│   └── tokenizers/                    # Serialized subword vocabularies & tokenizers
├── docs/
│   └── architecture.md                # System architecture and end-to-end dataflow
├── notebooks/
│   ├── 01_preprocessing_and_linguistics.ipynb
│   ├── 02_vectorization_and_embeddings.ipynb
│   ├── 03_probabilistic_and_sequence_models.ipynb
│   ├── 04_deep_learning_and_transformers.ipynb
│   └── 05_nlp_applications_and_deployment.ipynb
├── scripts/
│   ├── generate_dataset.py            # CLI synthetic dataset generator
│   ├── run_phase.py                   # Multi-phase runner CLI (--phase 1..5)
│   ├── run_pipeline.py                # End-to-end processing lifecycle runner
│   └── serve.py                       # FastAPI model server runner
├── src/
│   └── nlp_engine/
│       ├── config.py                  # Dynamic paths and configuration loader
│       ├── data/                      # Generator, loaders, and PyTorch dataset adapters
│       ├── preprocessing/             # Cleaning, Tokenizers, Morphology, Ambiguity
│       ├── representation/            # BoW, TF-IDF, Word2Vec, GloVe, BERT
│       ├── probabilistic/             # N-grams, HMM POS Tagger, Naive Bayes
│       ├── neural/                    # MLP, RNN, LSTM, GRU, Attention, Transformers
│       ├── applications/              # Intent Routing, Sentiment, NER, Hybrid Chatbot
│       ├── pipeline/                  # End-to-end Orchestrator & FastAPI REST application
│       └── utils/                     # Logger, Metrics calculations, Plotting tools
├── tests/                             # Comprehensive automated pytest suite
├── Makefile                           # Development automation commands
├── pyproject.toml                     # Setuptools packaging specification
├── requirements.txt                   # Dependency manifest
└── README.md                          # Project documentation
```

---

## 🔬 Core Engine Capabilities

| Phase | Focus Topic | Key Engine Modules |
| :--- | :--- | :--- |
| **Phase 1** | Text Processing & Linguistic Foundations | `nlp_engine.preprocessing` (`cleaner`, `tokenizer`, `morphology`, `ambiguity`) |
| **Phase 2** | Text Representation Techniques | `nlp_engine.representation` (`bow`, `tfidf`, `embeddings`, `contextual`) |
| **Phase 3** | NLP Models & Algorithms | `nlp_engine.probabilistic` (`ngram`, `hmm`, `classifier`) |
| **Phase 4** | Deep Learning in NLP | `nlp_engine.neural` (`mlp`, `rnn`, `attention`, `transformer`) |
| **Phase 5** | NLP Applications & Solutions | `nlp_engine.applications` (`ticket_classifier`, `sentiment`, `ner`, `chatbot`, `translation`) |

---

## 🚀 Quickstart & Setup

### 1. Initialize Virtual Environment with Python 3.12
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

### 2. Install Project Dependencies
```bash
uv pip install -e ".[dev]"
```

### 3. Generate Support Ticket Dataset
```bash
uv run python scripts/generate_dataset.py --samples 1000 --output data/raw/customer_tickets.csv
```

### 4. Run Any Milestone Phase
```bash
python scripts/run_phase.py --phase 1   # Phase 1: Preprocessing & Morphology
python scripts/run_phase.py --phase 2   # Phase 2: Text Representation & Embeddings
python scripts/run_phase.py --phase 3   # Phase 3: Probabilistic & Sequence Models
python scripts/run_phase.py --phase 4   # Phase 4: Deep Learning & Transformers
python scripts/run_phase.py --phase 5   # Phase 5: Applications & Deployment
```

### 5. Run Full End-to-End Pipeline
```bash
python scripts/run_pipeline.py
```

### 6. Run Automated Test Suite
```bash
pytest tests/ -v
```

### 7. Launch Serving REST API
```bash
python scripts/serve.py --host 0.0.0.0 --port 8000
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.
