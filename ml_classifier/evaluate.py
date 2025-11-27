import pickle
from dataset_loader import load_dataset
from preprocessing import clean_text
from sklearn.metrics import classification_report

with open("models/classifier.pkl", "rb") as f:
    obj = pickle.load(f)
    clf = obj["model"]
    embedder = obj["embedder"]

def evaluate():
    X_train, X_test, y_train, y_test = load_dataset()
    
    X_test_clean = [clean_text(x) for x in X_test]
    X_test_vec = embedder.encode(X_test_clean)

    pred= clf.predict(X_test_vec)
    
    print(classification_report(y_test, pred))


if __name__ == "__main__":
    evaluate()
    
