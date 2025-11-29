import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from classifier import RuleEmbeddingClassifier
from preprocessing import clean


def evaluate(dataset_path="ml_classifier/data/synthetic_ecommerce_data.csv"):
    df = pd.read_csv(dataset_path)
    df.dropna(inplace=True)

    texts = df["text"].tolist()
    labels = df["intent"].tolist()

    clf = RuleEmbeddingClassifier()

    predictions = []
    rule_hits = 0
    embed_hits = 0
    unknowns = 0

    for text, label in zip(texts, labels):

        # Check if rules match
        rule_intent = clf.match_rules(text)

        if rule_intent:
            if rule_intent == label:
                rule_hits += 1
            predictions.append(rule_intent)
            continue

        # Fallback to embeddings
        embed_intent = clf.embedding_fallback(text)

        if embed_intent == "unknown":
            unknowns += 1
        else:
            embed_hits += embed_intent == label

        predictions.append(embed_intent)

    # Print summary
    print("\n======== Evaluation Report ========")
    print(classification_report(labels, predictions))

    # Confusion matrix
    print("\n======== Confusion Matrix ========")
    print(confusion_matrix(labels, predictions))

    total = len(labels)

    print("\n======== Coverage Summary ========")
    print(f"Total samples: {total}")
    print(f"Rule-based matches: {rule_hits} ({rule_hits/total:.2%})")
    print(f"Embedding correct:  {embed_hits} ({embed_hits/total:.2%})")
    print(f"Unknown predictions: {unknowns} ({unknowns/total:.2%})")
    print(f"Overall accuracy: {(rule_hits + embed_hits)/total:.2%}")


if __name__ == "__main__":
    evaluate()
