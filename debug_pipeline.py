from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

model_dir = "model/distilbert-intent-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

clf = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=1)
out = clf("hello")
print(f"Output type: {type(out)}")
print(f"Output: {out}")
print(f"Output[0] type: {type(out[0])}")
print(f"Output[0]: {out[0]}")
