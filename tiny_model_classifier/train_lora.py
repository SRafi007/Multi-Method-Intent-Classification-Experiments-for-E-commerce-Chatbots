# scripts/train_lora.py
"""
Efficient fine-tuning with PEFT (LoRA). Requires `peft` and Transformers >= 4.33.
This lowers GPU memory and speeds up training for large models.
"""
import os
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_config, LoraConfig, get_peft_model
import numpy as np
from dataset_loader import load_csv
from preprocessing import clean_text

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "../model/distilbert-lora-intent"

def prepare():
    X_train, X_val, X_test, y_train, y_val, y_test = load_csv()
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

def main():
    ds, label_to_id, id_to_label = prepare()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    tokenized = ds.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_to_id))

    # LoRA config (tune these)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_lin", "v_lin", "k_lin", "fc"],  # may vary by architecture
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=50,
        fp16=True
    )

    def compute_metrics(p):
        from datasets import load_metric
        metric = load_metric("accuracy")
        preds = np.argmax(p.predictions, axis=1)
        acc = metric.compute(predictions=preds, references=p.label_ids)["accuracy"]
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import json
    with open(os.path.join(OUTPUT_DIR, "label_to_id.json"), "w") as f:
        json.dump(label_to_id, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "id_to_label.json"), "w") as f:
        json.dump(id_to_label, f, indent=2)

    print("LoRA training completed. Saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
