import pickle
from preprocessing import clean_text

with open("ml_classifier/models/classifier.pkl", "rb") as f:
    obj = pickle.load(f)
    clf = obj["model"]
    embedder = obj["embedder"]

def predict_intent(query:str)->str:
    cleaned = clean_text(query)
    vec = embedder.encode([cleaned])
    pred = clf.predict(vec)[0]
    return pred

if __name__ == "__main__":
    while True:
        q = input("User: ")
        print("intent: ", predict_intent(q))
    