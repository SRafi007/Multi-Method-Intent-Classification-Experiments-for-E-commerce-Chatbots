import pickle

import numpy as np 
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from dataset_loader import load_dataset
from preprocessing import clean_text    


def embed_sentences(model, sentences):
    sentences = [clean_text(sentence) for sentence in sentences]    
    return model.encode(sentences)

def train():
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_dataset()

    print("loading embedding model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding sentences...")
    X_train_vec = embed_sentences(embedder, X_train)
    X_test_vec = embed_sentences(embedder, X_test)

    print("Training model...")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_vec, y_train)

    print("Saving model...")
    with open("models/classifier.pkl", "wb") as f:
        pickle.dump({"model":clf, "embedder": embedder}, f)

    print("Model saved to models/classifier.pkl")

if __name__ == "__main__":
    train() 
    

    
    