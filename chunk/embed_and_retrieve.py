import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

device = 'cuda'
print(f"Using device: {device}")

#model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
model = "intfloat/e5-large-v2"

chunks = []
metadatas = []
with open("chunked_output.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        chunks.append(data["text"])
        metadatas.append({k: data[k] for k in data if k != "text"})

embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True, device=device)

def search(query, embeddings, top_k=3):
    query_emb = model.encode([query], convert_to_numpy=True, device=device)[0]
    sims = np.dot(embeddings, query_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-10)
    top_indices = np.argsort(sims)[::-1][:top_k]
    for idx in top_indices:
        print(f"\nScore: {sims[idx]:.3f}")
        print("Metadata:", metadatas[idx])
        print("Text:", chunks[idx][:500], "..." if len(chunks[idx]) > 1000 else "")
        print("-" * 40)


#query = "How do CD4 T cells respond to influenza infection?"
#search(query, embeddings, top_k=3)
