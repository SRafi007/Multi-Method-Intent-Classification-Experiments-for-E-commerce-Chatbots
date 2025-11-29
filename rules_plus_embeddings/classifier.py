import yaml
import re
import pickle
from sentence_transformers import SentenceTransformer
from preprocessing import clean


class RuleEmbeddingClassifier:
    def __init__(
        self,
        rules_path="rules_plus_embeddings/rules.yaml",
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
    ):
        # Load rules
        with open(rules_path, "r") as f:
            self.rules = yaml.safe_load(f)

        # Load embedding model
        self.embedder = SentenceTransformer(embed_model)

        # Build intent bank (embedding of each intent name)
        self.intent_labels = list(self.rules.keys())
        self.intent_vectors = self.embedder.encode(
            self.intent_labels, convert_to_numpy=True
        )

    def match_rules(self, query: str):
        text = clean(query)

        for intent, rule_set in self.rules.items():

            # Keyword match
            for kw in rule_set.get("keywords", []):
                if kw in text:
                    return intent

            # Regex match
            for pattern in rule_set.get("regex", []):
                if re.search(pattern, text):
                    return intent

        return None

    def embedding_fallback(self, query: str, threshold=0.45):
        q_vec = self.embedder.encode([clean(query)], convert_to_numpy=True)[0]

        # Compute cosine similarity
        similarities = (self.intent_vectors @ q_vec) / (
            (self.intent_vectors**2).sum(axis=1) ** 0.5 * (q_vec**2).sum() ** 0.5
        )

        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        if best_score >= threshold:
            return self.intent_labels[best_idx]

        return "unknown"

    def predict(self, query: str):
        # 1) Try rules first
        rule_intent = self.match_rules(query)
        if rule_intent:
            return rule_intent

        # 2) Fallback to embeddings
        return self.embedding_fallback(query)
