import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

from embedding_similarity.dataset_loader import load_dataset
from common.preprocessing import clean_text


def build_index():
    print("[1] Loading dataset...")
    examples, intents = load_dataset()

    print("[2] Cleaning examples...")
    cleaned = [clean_text(t) for t in examples]

    print("[3] Loading embedder (MiniLM)...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("[4] Embedding examples...")
    vectors = embedder.encode(cleaned, convert_to_numpy=True)

    print("[5] Saving index to disk...")
    np.save("embedding_similarity/embeddings/example_vectors.npy", vectors)
    np.save("embedding_similarity/embeddings/intents.npy", np.array(intents))
    with open("embedding_similarity/embeddings/embedder.pkl", "wb") as f:
        pickle.dump(embedder, f)

    print("Index built successfully!")


if __name__ == "__main__":
    build_index()
