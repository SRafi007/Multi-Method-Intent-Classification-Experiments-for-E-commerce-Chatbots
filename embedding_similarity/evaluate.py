"""
Evaluate embedding-similarity intent classifier.
Calculates accuracy, per-class metrics, confusion matrix,
and average inference latency.

Run:
    python evaluate.py --test data/intents.csv
"""

import argparse
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from embedding_similarity.dataset_loader import load_dataset
from common.preprocessing import clean_text
from embedding_similarity.inference import EmbeddingSimilarityRouter


def evaluate(test_csv: str):
    print("Loading evaluation dataset...")
    df = load_dataset(test_csv)
    df['text_clean'] = df['text'].apply(clean_text)

    router = EmbeddingSimilarityRouter()

    true_labels = []
    pred_labels = []
    latencies = []

    print(f"\nEvaluating {len(df)} samples...\n")

    for _, row in df.iterrows():
        text = row['text_clean']
        true = row['intent']

        # timing inference
        start = time.time()
        pred = router.route(text)
        latency = (time.time() - start) * 1000  # ms

        true_labels.append(true)
        pred_labels.append(pred)
        latencies.append(latency)

    # --- Metrics ---
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, digits=3)
    cm = confusion_matrix(true_labels, pred_labels)

    print("=== Evaluation Summary ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Avg inference time: {np.mean(latencies):.3f} ms")
    print("\n=== Classification Report ===")
    print(report)
    print("\n=== Confusion Matrix ===")
    print(cm)

    # Save results
    pd.DataFrame({
        "text": df["text"],
        "intent_true": true_labels,
        "intent_pred": pred_labels,
        "latency_ms": latencies
    }).to_csv("evaluation_results.csv", index=False)

    print("\nSaved detailed results → evaluation_results.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, required=True, help="Path to test CSV file")
    args = parser.parse_args()

    evaluate(args.test)


if __name__ == "__main__":
    main()
