import numpy as np
import pickle
from common.preprocessing import clean_text

# Preload everything for speed
example_vectors = np.load("embedding_similarity/embeddings/example_vectors.npy")
intents = np.load("embedding_similarity/embeddings/intents.npy")

with open("embedding_similarity/embeddings/embedder.pkl", "rb") as f:
    embedder = pickle.load(f)


def cosine_sim(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


def predict_intent(query: str, threshold=0.40):
    cleaned = clean_text(query)

    q_vec = embedder.encode([cleaned], convert_to_numpy=True)[0]

    # Compute cosine similarity with every training example
    sims = (
        example_vectors
        @ q_vec
        / (np.linalg.norm(example_vectors, axis=1) * np.linalg.norm(q_vec))
    )

    best_idx = sims.argmax()
    best_score = sims[best_idx]

    if best_score < threshold:
        return "unknown"

    return intents[best_idx]


if __name__ == "__main__":
    while True:
        q = input("User: ")
        print("Intent:", predict_intent(q))
