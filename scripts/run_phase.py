#!/usr/bin/env python3
"""
Multi-Phase Milestone Runner CLI.
Executes pipeline phases (Phase 1 to Phase 5) with full progress output.
"""

import argparse
from pathlib import Path
import sys
import numpy as np

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_engine.config import RAW_DATA_DIR, ARTIFACTS_DIR, PLOTS_DIR
from nlp_engine.data.generator import generate_support_tickets
from nlp_engine.preprocessing.cleaner import TextCleaner
from nlp_engine.preprocessing.tokenizer import WordTokenizer, SentenceTokenizer, BytePairEncoder
from nlp_engine.preprocessing.morphology import MorphologyProcessor
from nlp_engine.preprocessing.ambiguity import AmbiguityHandler
from nlp_engine.representation.bow import BagOfWords
from nlp_engine.representation.tfidf import TFIDFModel
from nlp_engine.representation.embeddings import Word2VecEngine
from nlp_engine.representation.contextual import ContextualEmbedder
from nlp_engine.probabilistic.ngram import NGramLanguageModel
from nlp_engine.probabilistic.hmm import HiddenMarkovModelTagger
from nlp_engine.probabilistic.classifier import TicketNaiveBayes
from nlp_engine.neural.mlp import TextMLPClassifier
from nlp_engine.neural.rnn import SequenceRNN
from nlp_engine.neural.attention import BahdanauAttention
from nlp_engine.neural.transformer import TransformerClassifier
from nlp_engine.applications.ticket_classifier import TicketClassifier
from nlp_engine.applications.sentiment import SentimentAnalyzer
from nlp_engine.applications.ner import CustomerNER
from nlp_engine.applications.chatbot import HybridSupportChatbot
from nlp_engine.applications.translation import SupportTranslator
from nlp_engine.utils.metrics import calculate_classification_metrics


def run_phase_1():
    print("=" * 80)
    print("PHASE 1: TEXT PROCESSING & LINGUISTIC FOUNDATIONS")
    print("=" * 80)
    cleaner = TextCleaner()
    word_tok = WordTokenizer()
    sent_tok = SentenceTokenizer()
    morph = MorphologyProcessor()
    ambig = AmbiguityHandler()

    sample_raw = "URGENT!! Billed $49 twice for order ORD-99214 on 2026-03-12. Please refund at once! Visit http://support.portal/help"
    print(f"\n[1] Raw Input Text:\n    '{sample_raw}'")
    cleaned = cleaner.clean(sample_raw)
    print(f"[2] Normalized & Cleaned Text:\n    '{cleaned}'")

    sentences = sent_tok.tokenize(sample_raw)
    words = word_tok.tokenize(cleaned)
    print(f"[3] Sentences ({len(sentences)}): {sentences}")
    print(f"[4] Word Tokens ({len(words)}): {words}")

    # Morphology comparison
    test_words = ["running", "orders", "billed", "charging", "delayed", "deliveries"]
    print("\n[5] Stemming (Porter) vs. Lemmatization (WordNet) Comparison:")
    comp = morph.compare_stem_vs_lemma(test_words)
    print(f"{'Word':12s} | {'Porter Stem':15s} | {'WordNet Noun':15s} | {'WordNet Verb':15s}")
    print("-" * 65)
    for c in comp:
        print(f"{c['word']:12s} | {c['stemmed']:15s} | {c['lemmatized_noun']:15s} | {c['lemmatized_verb']:15s}")

    # Subword BPE
    print("\n[6] Subword Byte-Pair Encoding (BPE):")
    bpe = BytePairEncoder(num_merges=30)
    bpe.fit(["order", "orders", "ordering", "ordered", "charge", "charged", "charging"])
    subwords = bpe.tokenize("charging orders")
    print(f"    Target: 'charging orders' -> Subwords: {subwords}")

    # Ambiguity
    print("\n[7] Linguistic Ambiguity Resolution:")
    res_verb = ambig.detect_syntactic_part_of_speech("book", "to", "a")
    res_noun = ambig.detect_syntactic_part_of_speech("book", "the", "carefully")
    print(f"    'to book a flight' -> POS: {res_verb}")
    print(f"    'the book carefully' -> POS: {res_noun}")


def run_phase_2():
    print("=" * 80)
    print("PHASE 2: TEXT VECTORIZATION & FEATURE REPRESENTATION")
    print("=" * 80)
    corpus = [
        "invoice charged twice for order ORD-101",
        "system crash throwing error 503 dashboard lag",
        "package delivery delayed for shipment tracking",
        "please refund excess subscription fee credit card"
    ]

    # 1. Bag of Words & Sparsity
    bow = BagOfWords()
    bow_matrix = bow.fit_transform(corpus)
    sparsity = bow.calculate_sparsity(bow_matrix)
    print(f"[1] Bag of Words Matrix Shape: {bow_matrix.shape}")
    print(f"    Vocabulary Size: {len(bow.feature_names_)}")
    print(f"    BoW Matrix Sparsity: {sparsity:.2f}%")

    # 2. TF-IDF
    tfidf = TFIDFModel()
    tfidf_matrix = tfidf.fit_transform(corpus)
    print(f"\n[2] TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    sample_doc = "invoice charged twice for order ORD-101"
    top_kw = tfidf.get_top_keywords(sample_doc, top_k=3)
    print(f"    Top TF-IDF Keywords for '{sample_doc}':")
    for term, weight in top_kw:
        print(f"      - {term:15s} : {weight:.4f}")

    # 3. Word2Vec
    print("\n[3] Word2Vec Distributed Semantic Embeddings:")
    tokenized = [doc.split() for doc in corpus]
    w2v = Word2VecEngine(vector_size=32, window=3, min_count=1)
    w2v.fit(tokenized)
    sim = w2v.similarity("invoice", "charged")
    print(f"    Cosine similarity between 'invoice' and 'charged': {sim:.4f}")

    # 4. Contextual Polysemy
    print("\n[4] Contextual Embeddings (Transformer Polysemy Intuition):")
    ctx = ContextualEmbedder()
    poly_res = ctx.demonstrate_polysemy(
        "charge",
        "Please refund the unexpected credit card charge.",
        "Plug in the device cable to charge the battery."
    )
    print(f"    Target word: {poly_res['target_word']}")
    print(f"    Cosine similarity between distinct contexts: {poly_res['cosine_similarity']:.4f}")
    print(f"    Polysemy shift detected: {poly_res['polysemy_detected']}")


def run_phase_3():
    print("=" * 80)
    print("PHASE 3: PROBABILISTIC MODELING & SEQUENCE TAGGING")
    print("=" * 80)
    df = generate_support_tickets(n_samples=300, random_state=42)

    # 1. N-Gram Language Model
    print("[1] N-Gram Language Model with Laplace (Add-1) Smoothing:")
    lm = NGramLanguageModel(n=2, alpha=1.0)
    tokenized_sents = [t.lower().split() for t in df["text"].tolist()[:50]]
    lm.fit(tokenized_sents)
    test_sent = ["please", "refund", "my", "order"]
    ppl = lm.calculate_sentence_perplexity(test_sent)
    print(f"    Sentence: {' '.join(test_sent)}")
    print(f"    Perplexity: {ppl:.2f}")

    # 2. HMM POS Tagger with Viterbi
    print("\n[2] Hidden Markov Model POS Tagging (Viterbi Algorithm):")
    tagged_corpus = [
        [("refund", "VERB"), ("the", "DET"), ("charge", "NOUN")],
        [("the", "DET"), ("charge", "NOUN"), ("failed", "VERB")],
        [("please", "ADV"), ("cancel", "VERB"), ("my", "PRON"), ("order", "NOUN")]
    ]
    hmm = HiddenMarkovModelTagger()
    hmm.fit(tagged_corpus)
    test_query = ["please", "refund", "the", "charge"]
    viterbi_tags = hmm.viterbi(test_query)
    print(f"    Decoded Sequence: {viterbi_tags}")

    # 3. Naive Bayes Ticket Routing
    print("\n[3] Multinomial Naive Bayes Ticket Routing Classifier:")
    tfidf = TFIDFModel(max_features=500)
    cleaner = TextCleaner()
    cleaned = cleaner.clean_batch(df["text"].tolist())
    X = tfidf.fit_transform(cleaned)
    y = df["category"].values

    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    nb = TicketNaiveBayes(alpha=1.0)
    nb.fit(X_train, y_train)
    preds = nb.predict(X_test)

    metrics = calculate_classification_metrics(y_test, preds)
    print(f"    Accuracy       : {metrics['accuracy']*100:.2f}%")
    print(f"    Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"    Macro Recall   : {metrics['macro_recall']:.4f}")
    print(f"    Macro F1-Score : {metrics['macro_f1']:.4f}")


def run_phase_4():
    print("=" * 80)
    print("PHASE 4: DEEP LEARNING & SEQUENCE ARCHITECTURES")
    print("=" * 80)

    # 1. MLP Classifier
    print("[1] PyTorch Multi-Layer Perceptron (MLP) Baseline:")
    mlp = TextMLPClassifier(input_dim=50, hidden_dims=[64, 32], num_classes=4)
    dummy_x = np.random.randn(100, 50).astype("float32")
    dummy_y = np.random.randint(0, 4, size=100)
    losses = mlp.fit(dummy_x, dummy_y, epochs=3, batch_size=32)
    print(f"    Trained 3 Epochs -> Final Loss: {losses[-1]:.4f}")

    # 2. Vanishing Gradient Diagnostic in Sequence RNN
    print("\n[2] Sequence RNN & Vanishing Gradient Diagnostic:")
    seq_rnn = SequenceRNN(vocab_size=1000, hidden_dim=64, num_classes=4)
    diag = seq_rnn.diagnose_vanishing_gradient(sequence_length=40)
    print(f"    Vanilla RNN Gradient Norm : {diag['vanilla_rnn_grad_norm']:.6e}")
    print(f"    LSTM Gradient Norm        : {diag['lstm_grad_norm']:.6e}")
    if "ratio_lstm_to_rnn" in diag:
        print(f"    LSTM / RNN Gradient Ratio : {diag['ratio_lstm_to_rnn']:.2f}x (Gradient retention)")

    # 3. Transformer Encoder Block
    print("\n[3] Transformer Encoder Architecture:")
    transformer = TransformerClassifier(vocab_size=1000, d_model=64, nhead=2, num_classes=4)
    print("    Multi-Head Attention & Positional Encoding initialized successfully.")


def run_phase_5():
    print("=" * 80)
    print("PHASE 5: FULL NLP APPLICATION SUITE & INFERENCE")
    print("=" * 80)
    df = generate_support_tickets(n_samples=300, random_state=42)

    # 1. Ticket Classifier
    print("[1] End-to-End Ticket Classifier & Triage:")
    triage_clf = TicketClassifier()
    triage_clf.fit(df)
    test_ticket = "URGENT: Billed $49 twice for order ORD-44102 on 2026-04-10. Refund immediately!"
    res = triage_clf.predict_ticket(test_ticket)
    print(f"    Inquiry: '{test_ticket}'")
    print(f"    Category: {res['category']} (conf: {res['category_confidence']})")
    print(f"    Urgency: {res['urgency']} | Is Spam: {res['is_spam']}")

    # 2. Sentiment & Aspect Analysis
    print("\n[2] Sentiment & Opinion Mining:")
    sentiment_engine = SentimentAnalyzer()
    sent_res = sentiment_engine.analyze_polarity("The delivery courier arrived fast, but the billing fee was awful!")
    aspects = sentiment_engine.analyze_aspects("The delivery courier arrived fast, but the billing fee was awful!")
    print(f"    Overall: {sent_res['sentiment']} (polarity: {sent_res['polarity']})")
    print(f"    Aspects: {aspects}")

    # 3. Customer NER
    print("\n[3] Customer Support Named Entity Recognition (NER):")
    ner = CustomerNER()
    entities = ner.extract_entities(test_ticket)
    print(f"    Extracted Entities ({len(entities)}):")
    for ent in entities:
        print(f"      - [{ent['label']:10s}]: '{ent['entity']}'")

    # 4. Dual-Mode Chatbot
    print("\n[4] Dual-Mode Conversational Chatbot:")
    bot = HybridSupportChatbot()
    bot_reply_1 = bot.respond("How can I reset my password?")
    bot_reply_2 = bot.respond("Where is my package for order ORD-123?")
    print(f"    User: 'How can I reset my password?' -> [{bot_reply_1['mode']}] {bot_reply_1['response']}")
    print(f"    User: 'Where is my package?'         -> [{bot_reply_2['mode']}] {bot_reply_2['response']}")

    # 5. Translation & BLEU Score
    print("\n[5] Multilingual Support & BLEU Evaluation:")
    translator = SupportTranslator()
    eval_res = translator.evaluate_translation(
        reference="please refund my order immediately",
        hypothesis="please refund my order now"
    )
    print(f"    BLEU Score on Support Translation: {eval_res['bleu_score']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run NLP Engine Pipeline Phases.")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5], required=True, help="Milestone phase (1 to 5)")
    args = parser.parse_args()

    phase_map = {
        1: run_phase_1,
        2: run_phase_2,
        3: run_phase_3,
        4: run_phase_4,
        5: run_phase_5,
    }
    phase_map[args.phase]()


if __name__ == "__main__":
    main()
