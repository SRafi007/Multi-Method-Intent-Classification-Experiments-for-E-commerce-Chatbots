# scripts/inference.py
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from common.preprocessing import clean_text

MODEL_DIR = "model/distilbert-intent-classifier"  # or distilbert-lora-intent

def load_model(model_dir=MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    # create a text-classification pipeline
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=1)
    # load id->label map if exists
    id_to_label_path = os.path.join(model_dir, "id_to_label.json")
    if os.path.exists(id_to_label_path):
        with open(id_to_label_path, "r") as f:
            id_to_label = json.load(f)
    else:
        id_to_label = None
    return clf, id_to_label

def predict(clf, text: str, id_to_label=None):
    text = clean_text(text)
    out = clf(text)[0][0]  # e.g. {'label': 'LABEL_0', 'score': 0.98}
    label = out["label"]
    score = out["score"]
    # If id_to_label mapping uses integers:
    if id_to_label and label.startswith("LABEL_"):
        idx = int(label.split("_")[-1])
        label = id_to_label.get(str(idx), label)
    return {"intent": label, "score": float(score)}

if __name__ == "__main__":
    clf, id_to_label = load_model()
    while True:
        q = input("User: ")
        res = predict(clf, q, id_to_label=id_to_label)
        print(f"Intent: {res['intent']} (score={res['score']:.3f})")
