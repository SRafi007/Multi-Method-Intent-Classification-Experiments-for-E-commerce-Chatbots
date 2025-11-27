
# Multi-Method Intent Classification  for E-commerce Chatbots

This project demonstrates **five different approaches** to classify user intents for an e-commerce chatbot.  
Each method is isolated in its own directory so you can **run, test, and compare** them independently.

The goal is to understand how each technique performs in terms of **speed, accuracy, flexibility, and maintenance**.

---

##  What’s Inside

### 1. **ML Classifier + Embeddings (Fastest)**
- Logistic Regression / SVM / XGBoost  
- MiniLM embeddings  
- CPU inference <5ms  

### 2. **Rules + Embeddings (High Precision)**
- Keyword / regex rules  
- Embedding fallback for fuzzy inputs  

### 3. **Embedding Similarity Routing (Zero Maintenance)**
- No training required  
- Intent bank + cosine similarity  

### 4. **Tiny Model Classifier (Most Flexible)**
- DistilBERT / MiniLM / Phi-mini  
- Small model fine-tuned for intents  

### 5. **Embedding-Based Routing (Best Overall Balance)**
- Production-friendly router  
- Threshold-based scoring  

---

## 📂 Project Structure

```

intent-lab/
│
├── data/
├── common/
├── 1_ml_classifier/
├── 2_rules_plus_embeddings/
├── 3_embedding_similarity/
├── 4_tiny_model_classifier/
├── 5_embedding_based_routing/
└── benchmarks/

````

Each folder includes its **own pipelines** for training, inference, and evaluation.

---

##  Evaluation & Benchmarking

Includes utilities to measure:
- Accuracy / F1 score  
- Latency per method  
- Confusion matrix  

Run benchmarks to compare how each classifier behaves in real-time scenarios.

---

##  Why This Project?

E-commerce chatbots often struggle with tricky, ambiguous queries.  
This repository provides a **practical, hands-on comparison** of multiple approaches so developers can pick the right method for their system.

Whether you care about speed, accuracy, or zero maintenance — you’ll find a working solution here.

---

## 🛠️ Getting Started

Clone the repo:

```bash
git clone https://github.com/your-username/intent-lab.git
cd intent-lab
````

Install dependencies:

```bash
pip install -r requirements.txt
```

Train & test any module independently.
Example (ML classifier):

```bash
cd 1_ml_classifier
python train.py
python inference.py
```

---

## 🤝 Contributions

PRs and improvements are welcome.
This repo is meant to grow as new intent-classification techniques emerge.

---

##  If You Like This Project

Give it a star on GitHub and share it!
It helps more developers learn and build smarter assistants.

```
