import json
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

DATA_PATH = "../../data/pqa_artificial.jsonl"
MODEL_NAME = "intfloat/e5-small-v2"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "pqa_artificial"
EMBED_DIM = 384
BATCH_SIZE = 16
PROGRESS_FILE = "../../data/embedding_progress_for_pqa_art.log"


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
    Returns -1 if the file doesn't exist or is empty, starting from the beginning.
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
    """Saves the ID of the last successfully processed article to the progress file."""
    try:
        with open(PROGRESS_FILE, "w") as f:
            f.write(str(last_id))
    except IOError as e:
        print(f"Critical Error: Could not write to progress file '{PROGRESS_FILE}'. Exiting. Error: {e}")
        sys.exit(1)

def read_data(path, start_from_id=-1):
    """
    Data needs to be read line by line.
    It yields the line number (ID) and the parsed JSON object.
    It skips lines that have already been processed based on `start_from_id`.
    """
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i <= start_from_id:
                continue
            yield i, json.loads(line)

def generate_embeddings_in_batches(texts, model, tokenizer, device):
    """
    Generates embeddings for a list of texts in a single batch.
    This is much more efficient than embedding one by one.
    """
    inputs = [f"passage: {text}" for text in texts]
    encoded = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        out = model(**encoded)
    embeddings = out.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

def main():
    """Main function to run the embedding and upsert process."""
    model, tokenizer, device = initialize()
    client = qdrant_setup()

    last_processed_id = get_last_processed_id()
    print("-" * 50)
    if last_processed_id > -1:
        print(f"Resuming embedding process from after article ID: {last_processed_id}")
    else:
        print("Starting new embedding process from the beginning.")
    print("-" * 50)

    try:
        print("Counting total articles in the dataset...")
        total_lines = sum(1 for _ in open(DATA_PATH, 'r', encoding='utf-8'))
        print(f"Found {total_lines} total articles.")
    except FileNotFoundError:
        print(f"Error: The file was not found at {DATA_PATH}")
        return

    article_batch_for_model = []
    id_batch_for_qdrant = []
    payload_batch_for_qdrant = []

    article_generator = read_data(DATA_PATH, start_from_id=last_processed_id)
    
    with tqdm(article_generator, total=total_lines, initial=last_processed_id + 1, desc="Embedding Knowledge only") as pbar:
        for idx, article in pbar:
            knowledge = article.get('Knowledge', '').strip()
            text_to_embed = f"{knowledge}"

            # Add data to the current batch
            article_batch_for_model.append(text_to_embed)
            id_batch_for_qdrant.append(idx)
            payload_batch_for_qdrant.append(article)

            # When the batch is full, process it
            if len(article_batch_for_model) >= BATCH_SIZE:
                embeddings = generate_embeddings_in_batches(article_batch_for_model, model, tokenizer, device)

                points = [
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload=payload
                    ) for point_id, vector, payload in zip(id_batch_for_qdrant, embeddings, payload_batch_for_qdrant)
                ]
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                    wait=True
                )
                log_progress(id_batch_for_qdrant[-1])
                article_batch_for_model = []
                id_batch_for_qdrant = []
                payload_batch_for_qdrant = []

    if article_batch_for_model:
        print(f"Processing the final batch of {len(article_batch_for_model)} articles...")
        embeddings = generate_embeddings_in_batches(article_batch_for_model, model, tokenizer, device)

        points = [
            qdrant_models.PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload
            ) for point_id, vector, payload in zip(id_batch_for_qdrant, embeddings, payload_batch_for_qdrant)
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        log_progress(id_batch_for_qdrant[-1])
        print("Final batch processed.")

    print("-" * 50)
    print("Embedding process completed successfully.")
    print("-" * 50)


if __name__ == "__main__":
    main()