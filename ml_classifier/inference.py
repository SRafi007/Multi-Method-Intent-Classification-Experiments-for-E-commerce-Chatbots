import pickle
from preprocessing import clean_text

with open("models/classifier.pkl", "rb") as f:
    model = pickle.load(f)
    clf = obj["model"]
    embedder = obj["embedder"]

def predict_intent(query:str)->str:
    cleaned = clean_text(query)
    vec = embedder.encode([cleaned])
    pred = clf.redict(vec)[0]
    return pred

if __name__ == "__main__":
    while True:
        q = intent("User: ")
        print("intent: ", predict_intent(q))
    