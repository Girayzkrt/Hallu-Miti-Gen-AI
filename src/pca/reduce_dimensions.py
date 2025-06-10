import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.decomposition import PCA
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from tqdm import tqdm

MODEL_NAME = "intfloat/e5-large-v2"
JSON_PATH = "../data/parsed_pmc_2_1000.jsonl"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "pubmed_articles"
TARGET_DIM = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
if torch.cuda.is_available():
    model.cuda()

def get_embedding(text):
    inp = f"passage: {text.strip()}"
    encoded = tokenizer(inp, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        encoded = {k: v.cuda() for k, v in encoded.items()}
    with torch.no_grad():
        out = model(**encoded)
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

df = pd.read_json(JSON_PATH, lines=True)
df = df.dropna(subset=["title", "abstract"])
texts = (df["title"] + ". " + df["abstract"]).tolist()

print("Generating embeddings...")
embeddings = np.array([get_embedding(text) for text in tqdm(texts)])

print("Running PCA...")
pca = PCA(n_components=TARGET_DIM)
reduced_embeddings = pca.fit_transform(embeddings)

print("Uploading to Qdrant...")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=TARGET_DIM, distance=Distance.COSINE)
)

points = [
    PointStruct(id=i, vector=reduced_embeddings[i], payload={"text": texts[i]})
    for i in range(len(reduced_embeddings))
]

client.upload_points(collection_name=COLLECTION_NAME, points=points)

print("Done.")
