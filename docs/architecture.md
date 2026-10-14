# System Architecture: End-to-End Intelligent Customer Support & Sentiment Analytics

> **Domain:** Customer Support Operations, Issue Categorization, Automated Triage, and Conversational Assistance  
> **Package:** `nlp_engine`

---

## 🏗️ Architectural Overview

The system processes incoming customer communications (emails, live chat messages, tickets, reviews) through five sequential stages, transforming raw unstructured natural language into structured analytical insights, automated ticket resolutions, and routing decisions.

```mermaid
flowchart TD
    subgraph Ingestion ["Ingestion & Synthesis"]
        A["Incoming Ticket / Query"] --> B["Synthetic Data Generator<br/>(scripts/generate_dataset.py)"]
    end

    subgraph Stage1 ["Text Processing & Morphology"]
        B & A --> C["TextCleaner<br/>(Noise, URLs, Punctuation)"]
        C --> D["Tokenizer<br/>(Word, Sentence, BPE Subword)"]
        D --> E["MorphologyProcessor<br/>(Porter Stem vs. WordNet Lemma)"]
        E --> F["AmbiguityHandler<br/>(POS-guided Disambiguation)"]
    end

    subgraph Stage2 ["Text Representation"]
        F --> G1["BagOfWords<br/>(Sparse Matrix)"]
        F --> G2["TF-IDF Engine<br/>(Mathematical Weights)"]
        F --> G3["Word2Vec / GloVe<br/>(Dense Embeddings)"]
        F --> G4["Contextual Embedder<br/>(Transformer / BERT)"]
    end

    subgraph Stage3 ["Probabilistic & Sequence"]
        G1 & G2 --> H1["Naive Bayes Classifier<br/>(Multinomial Routing)"]
        F --> H2["N-gram Language Model<br/>(Laplace Smoothing & Perplexity)"]
        F --> H3["HMM POS Tagger<br/>(Viterbi Algorithm)"]
    end

    subgraph Stage4 ["Deep Learning Sequence Models"]
        G3 & G4 --> I1["Feedforward MLP"]
        G3 --> I2["Recurrent Networks<br/>(RNN / LSTM / GRU)"]
        I2 --> I3["Attention Mechanism<br/>(Bahdanau / Luong)"]
        G4 --> I4["Transformer Encoder Block"]
    end

    subgraph Stage5 ["Applications & Deployment"]
        H1 & I4 --> J1["Ticket Classifier<br/>(Urgency & Category)"]
        F & G2 --> J2["Sentiment Engine<br/>(Polarity & Opinion Mining)"]
        F --> J3["Customer NER<br/>(Order ID, Products, Names)"]
        J1 & J3 --> J4["Hybrid Chatbot<br/>(Rules + ML Fallback)"]
        F --> J5["Translation Layer<br/>(Multilingual BLEU)"]
    end

    subgraph Serving ["Production API"]
        J1 & J2 & J3 & J4 & J5 --> K["FastAPI Service<br/>(src/nlp_engine/pipeline/api.py)"]
    end
```

---

## 📦 Data Schemas & Contracts

### 1. Ingestion Record Schema (`data/raw/customer_tickets.csv`)
| Field | Type | Description |
| :--- | :--- | :--- |
| `ticket_id` | String | Unique identifier (e.g., `TCK-10492`) |
| `customer_name` | String | Name of customer |
| `email` | String | Customer contact address |
| `subject` | String | Brief ticket summary |
| `body` | String | Full message / inquiry text |
| `category` | String | Target category (e.g., `Billing & Invoicing`, `Technical Support`) |
| `urgency` | String | Priority level (`Low`, `Medium`, `High`, `Critical`) |
| `sentiment` | String | Ground truth polarity (`Positive`, `Neutral`, `Negative`) |
| `is_spam` | Integer | Binary flag (`0` legitimate, `1` spam) |
| `entities` | JSON/String | Extracted entities dictionary (`order_id`, `product`, `date`) |

### 2. FastAPI Request & Response Contracts (`src/nlp_engine/pipeline/api.py`)
- **`POST /api/v1/predict/ticket`**: Classifies category, urgency, spam risk, and sentiment.
- **`POST /api/v1/sentiment`**: Extracts sentence-level polarity and subjective tone.
- **`POST /api/v1/ner`**: Extracts customer entities (`ORDER_ID`, `PRODUCT`, `DATE`, `PERSON`).
- **`POST /api/v1/chat`**: Generates a conversational resolution using rule tree or ML intent matching.
- **`GET /health`**: Microservice health, model status, and version telemetry.
