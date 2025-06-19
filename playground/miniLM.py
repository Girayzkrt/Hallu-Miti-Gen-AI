import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from nltk.tokenize import sent_tokenize
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 480
OVERLAP_SENTENCES = 2
BATCH_SIZE = 128
JSONL_PATH = "../../data/parsed_pmc_2_1000.jsonl"
QDRANT_COLLECTION = "minilm_chunks"
EMBEDDING_DIM = 384
model = SentenceTransformer(MODEL_NAME)
tokenizer = model.tokenizer

qdrant = QdrantClient("localhost", port=6333)

if QDRANT_COLLECTION not in [c.name for c in qdrant.get_collections().collections]:
    print("Creating new vector collection")
    qdrant.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=True))

def chunk_document(doc, doc_idx):
    for field in doc:
        if not doc[field] or not isinstance(doc[field], str):
            continue
        sentences = sent_tokenize(doc[field])
        i = 0
        chunk_id = 0
        last_chunk_text = None
        while i < len(sentences):
            chunk_sentences = []
            tokens = 0
            if chunk_id > 0:
                overlap_start = max(i - OVERLAP_SENTENCES, 0)
                chunk_sentences.extend(sentences[overlap_start:i])
                tokens = count_tokens(" ".join(chunk_sentences))
            while i < len(sentences) and tokens < MAX_TOKENS:
                next_sentence = sentences[i]
                next_tokens = count_tokens(next_sentence)
                if tokens + next_tokens > MAX_TOKENS and chunk_sentences:
                    break
                chunk_sentences.append(next_sentence)
                tokens += next_tokens
                i += 1
            chunk_text = " ".join(chunk_sentences)
            tokenized = tokenizer.encode(chunk_text, add_special_tokens=True)
            if len(tokenized) > 512:
                chunk_text = tokenizer.decode(tokenized[:512], skip_special_tokens=True)
            if chunk_text == last_chunk_text:
                break
            last_chunk_text = chunk_text
            yield {
                "original_index": doc_idx,
                "section": field,
                "chunk_id": chunk_id,
                "text": chunk_text,
            }
            chunk_id += 1

def get_last_point_id(qdrant, collection):
    """Get highest point id stored in Qdrant collection."""
    try:
        hits = qdrant.scroll(collection, limit=1, with_payload=True, order="desc", offset=0)
        if hits and hits[0]:
            return hits[0][0].id
        else:
            return -1
    except Exception:
        return -1

def get_already_indexed_ids(qdrant, collection):
    """Get all indexed point ids as a set"""
    all_ids = set()
    offset = 0
    limit = 10000
    while True:
        points, _ = qdrant.scroll(collection, limit=limit, offset=offset, with_payload=True)
        if not points:
            break
        for point in points:
            all_ids.add(point.payload.get('source_id'))
        offset += limit
    return all_ids

def make_source_id(doc_idx, chunk_id):
    return int(f"{doc_idx:08}{chunk_id:04}")

already_indexed = set()

with open(JSONL_PATH, "r", encoding="utf-8") as fin:
    batch = []
    meta_batch = []
    id_batch = []
    doc_idx = 0

    for line in fin:
        doc = json.loads(line)
        for chunk in chunk_document(doc, doc_idx):
            source_id = make_source_id(chunk['original_index'], chunk['chunk_id'])
            if source_id in already_indexed:
                continue
            batch.append(chunk["text"])
            meta = {
                "original_index": chunk["original_index"],
                "section": chunk["section"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source_id": source_id
            }
            meta_batch.append(meta)
            id_batch.append(source_id)
            if len(batch) >= BATCH_SIZE:
                vectors = model.encode(batch, show_progress_bar=False, batch_size=BATCH_SIZE, normalize_embeddings=False)
                points = [
                    PointStruct(id=id_batch[i], vector=vectors[i].tolist(), payload=meta_batch[i])
                    for i in range(len(batch))
                ]
                qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
                print(f"Indexed up to doc {meta_batch[-1]['original_index']}, chunk {meta_batch[-1]['chunk_id']}")
                batch, meta_batch, id_batch = [], [], []
        doc_idx += 1

    if batch:
        vectors = model.encode(batch, show_progress_bar=False, batch_size=BATCH_SIZE, normalize_embeddings=False)
        points = [
            PointStruct(id=id_batch[i], vector=vectors[i].tolist(), payload=meta_batch[i])
            for i in range(len(batch))
        ]
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"Final batch indexed up to doc {meta_batch[-1]['original_index']}, chunk {meta_batch[-1]['chunk_id']}")
