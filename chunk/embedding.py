import json
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/e5-large-v2') ## need to check PubMebBERT saved bookmark on chrome
model = model.to('cuda')

path = 'chunked_output.jsonl'
texts = []

with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        texts.append(data['text'])

embeddings = []

for text in tqdm(texts, desc="Embedding texts"):
    embedding = model.encode(
        text,
        batch_size=32,
        device='cuda',
        show_progress_bar=True,
        convert_to_numpy=True
        )

np.save('embeddings.npy', np.array(embeddings))
