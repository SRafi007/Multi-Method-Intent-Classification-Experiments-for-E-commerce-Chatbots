
# **PHASE 1 — TRAINING FLOW**

You run:

```
python ml_classifier/train.py
```

---

# **PHASE 2 — INFERENCE FLOW**

Now run:

```
python ml_classifier/inference.py
```

You type a message:

```
User: Where is my order?
```

---

# **PHASE 3 — EVALUATION FLOW**

Run:

```
python ml_classifier/evaluate.py
```


---

# **Summary of Full Run Flow**

```
TRAINING
--------
CSV → load → clean → embed → train ML → save model

INFERENCE
---------
user text → clean → embed → classifier.predict → intent

EVALUATION
----------
test data → clean → embed → predict → generate metrics
```

