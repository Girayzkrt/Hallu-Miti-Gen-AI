import json
import torch
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

def load_questions(jsonl_path):
    questions = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data["Question"])
    return questions

embedder = SentenceTransformer("intfloat/e5-small-v2", device="cuda" if torch.cuda.is_available() else "cpu")

qdrant = QdrantClient(host="localhost", port=6333)
collection_name = "pmc_chunked_title_abstract"

questions = load_questions("../../data/json_files/pqa_labeled.jsonl")

results = []

for q_idx, question in enumerate(tqdm(questions, desc="Processing Questions")):
    q_vector = embedder.encode(question).tolist()
    search_result = qdrant.search(
        collection_name=collection_name,
        query_vector=q_vector,
        limit=5,
        with_payload=True
    )

    for rank, hit in enumerate(search_result):
        result = {
            "Question_ID": q_idx + 1,
            "Question": question,
            "Retrieved_Title": hit.payload.get("title", ""),
            "Retrieved_Abstract": hit.payload.get("abstract", ""),
            "Similarity_Score": hit.score,
            "Rank": rank + 1
        }
        results.append(result)

df = pd.DataFrame(results)
df.to_csv("../../data/eval_reports/retrieval_similarity_results(with_chunk).csv", index=False)
print("Saved results'")
