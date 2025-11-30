# scripts/train.py
import os
from datasets import Dataset, DatasetDict, ClassLabel, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from dataset_loader import load_csv
from preprocessing import clean_text
from typing import Dict

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "../model/distilbert-intent-classifier"

def prepare_datasets():
    X_train, X_val, X_test, y_train, y_val, y_test = load_csv()
    # clean
    X_train = [clean_text(x) for x in X_train]
    X_val = [clean_text(x) for x in X_val]
    X_test = [clean_text(x) for x in X_test]

    labels = sorted(list(set(y_train + y_val + y_test)))
    label_to_id = {l: i for i, l in enumerate(labels)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    def to_dataset(X, y):
        return Dataset.from_dict({"text": X, "label": [label_to_id[l] for l in y]})

    ds = DatasetDict({
        "train": to_dataset(X_train, y_train),
        "validation": to_dataset(X_val, y_val),
        "test": to_dataset(X_test, y_test)
    })

    return ds, label_to_id, id_to_label

def tokenize_and_map(ds, tokenizer):
    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    return ds.map(preprocess, batched=True)

def compute_metrics(p):
    metric = load_metric("accuracy")
    preds = np.argmax(p.predictions, axis=1)
    acc = metric.compute(predictions=preds, references=p.label_ids)["accuracy"]
    return {"accuracy": acc}

def main():
    ds, label_to_id, id_to_label = prepare_datasets()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized = tokenize_and_map(ds, tokenizer)
    num_labels = len(label_to_id)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        logging_steps=50,
        fp16=False  # set True if GPU with mixed precision available
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    # Save final model and label maps
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # write label maps
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "label_to_id.json"), "w") as f:
        import json; json.dump(label_to_id, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "id_to_label.json"), "w") as f:
        import json; json.dump(id_to_label, f, indent=2)

    print("Training finished. Model saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
