import json
import os
import sys
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

DATA_PATH = "../../data/json_files/parsed_pmc_2.jsonl"
MODEL_NAME = "intfloat/e5-small-v2"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "pmc_chunked_title_abstract"
EMBED_DIM = 384
BATCH_SIZE = 16
CHUNK_SIZE = 480
CHUNK_OVERLAP = 50
PROGRESS_FILE = "../../data/embedding_progress.log"


def initialize():
    """Initializes and returns the model, tokenizer, and device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return model, tokenizer, device

def qdrant_setup():
    """Initializes the Qdrant client and ensures the collection exists."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' not found. Creating new collection.")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(
                size=EMBED_DIM,
                distance=qdrant_models.Distance.COSINE
            ),
        )
        print("Collection created.")
    return client

def get_last_processed_id():
    """
    Reads the ID of the last successfully embedded article from the progress file.
    """
    if not os.path.exists(PROGRESS_FILE):
        return -1
    try:
        with open(PROGRESS_FILE, "r") as f:
            content = f.read().strip()
            return int(content) if content else -1
    except (IOError, ValueError):
        print(f"Warning: Could not read progress file '{PROGRESS_FILE}'. Starting from scratch.")
        return -1

def log_progress(last_id):
    """
    Saves the ID of the last successfully processed article to the progress file.
    """
    try:
        with open(PROGRESS_FILE, "w") as f:
            f.write(str(last_id))
    except IOError as e:
        print(f"Critical Error: Could not write to progress file '{PROGRESS_FILE}'. Exiting. Error: {e}")
        sys.exit(1)

def read_data(path, start_from_id=-1):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i <= start_from_id:
                continue
            yield i, json.loads(line)

def embed_text(text, model, tokenizer, device, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Embeds a single text by chunking if it exceeds `chunk_size` tokens, using overlap.
    """
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) <= chunk_size:
        input_ids = torch.tensor([tokens], device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state.mean(dim=1).cpu().numpy()[0]
    

    print(f"Text is {len(tokens)} tokens long - {chunk_size} – splitting")
    embeddings = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        chunk = tokens[start : start + chunk_size]
        input_ids = torch.tensor([chunk], device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        emb = out.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        embeddings.append(emb)
    return np.mean(embeddings, axis=0)

def main():
    """Main function to run the embedding and upsert process."""
    model, tokenizer, device = initialize()
    client = qdrant_setup()

    last_processed_id = get_last_processed_id()
    print("-" * 50)
    if last_processed_id > -1:
        print(f"Resuming embedding process from after article ID: {last_processed_id}")
    else:
        print("Sil bastann baslamak gerek bazen")
    print("-" * 50)

    try:
        print("Counting total articles in the dataset...")
        total_lines = sum(1 for _ in open(DATA_PATH, 'r', encoding='utf-8'))
        print(f"Found {total_lines} total articles.")
    except FileNotFoundError:
        print(f"Error: The file was not found at {DATA_PATH}")
        return

    id_batch = []
    payload_batch = []
    embeddings_batch = []

    article_generator = read_data(DATA_PATH, start_from_id=last_processed_id)
    with tqdm(article_generator, total=total_lines, initial=last_processed_id + 1, desc="Embedding Articles") as pbar:
        for idx, article in pbar:
            title = article.get('title', '').strip()
            abstract = article.get('abstract', '').strip()
            text_to_embed = f"{title}. {abstract}"

            vec = embed_text(text_to_embed, model, tokenizer, device)

            id_batch.append(idx)
            payload_batch.append(article)
            embeddings_batch.append(vec)

            if len(embeddings_batch) >= BATCH_SIZE:
                points = [
                    qdrant_models.PointStruct(
                        id=pid,
                        vector=vec.tolist(),
                        payload=pl
                    ) for pid, vec, pl in zip(id_batch, embeddings_batch, payload_batch)
                ]
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                    wait=True
                )
                log_progress(id_batch[-1])
                id_batch, payload_batch, embeddings_batch = [], [], []

 
    if embeddings_batch:
        print(f"Processing the final batch of {len(embeddings_batch)} docs")
        points = [
            qdrant_models.PointStruct(
                id=pid,
                vector=vec.tolist(),
                payload=pl
            ) for pid, vec, pl in zip(id_batch, embeddings_batch, payload_batch)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        log_progress(id_batch[-1])
        print("Final batch processed.")

    print("-" * 50)
    print("Done")
    print("-" * 50)


if __name__ == "__main__":
    main()
