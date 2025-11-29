# Embedding Similarity Intent Router

This module uses vector similarity to detect intents without training any ML model.

## How it works
1. Load your dataset (text, intent)
2. Clean and embed all examples
3. Save embeddings as an index
4. At inference:
   - Embed query
   - Compute cosine similarity with all examples
   - Return the intent of the closest example

## Steps

### Build index
