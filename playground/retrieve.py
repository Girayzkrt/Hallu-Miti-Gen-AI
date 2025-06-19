import json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import pipeline
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import pandas as pd

MODEL_NAME = "intfloat/e5-large-v2"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "pubmed_articles"
EMBED_DIM = 1024

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

#Connection
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

query = "What are the comparative outcomes of Tenon duplication versus dura mater covering techniques in Ahmed glaucoma valve implantation?"

query_emb = get_embedding(query)

#search
hits = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_emb.tolist(),
    limit=1
)

#retrieve context
context_chunks = []
for hit in hits:
    payload = hit.payload
    title = payload.get("title", "")
    abstract = payload.get("abstract", "")
    results = payload.get("results", "")
    context_chunks.append(f"Title: {title}\nAbstract: {abstract}\nResults: {results}")

retrieved_context = "\n\n".join(context_chunks)
print("Retrieved context:", retrieved_context)

messages_with_context = [
    {
        "role": "system",
        "content": (
            "You are a board-certified medical expert. "
            "Answer each question accurately and concisely in 3-5 sentences. Use the provided context to inform your answer. Summarize the key points from the context that are relevant to the question. "
            "If the context is insufficient, state that you cannot answer based on the provided information. "
            "If the question is not related to medical topics, politely decline to answer."
        ),
    },
    {
        "role": "system",
        "content": f"Relevant research context:\n{retrieved_context}"
    },
    {
        "role": "user",
        "content": query,
    },
]

model_name = "Qwen/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt = tokenizer.apply_chat_template(messages_with_context, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
print("Token count:", len(tokenizer(prompt)["input_ids"]))
print("Prompt length:", len(prompt))
outputs = model.generate(
    **inputs,
    max_new_tokens=500,
    do_sample=False,
    temperature=0.1,
    top_p=0.95
)
generated = outputs[0][inputs["input_ids"].shape[1]:]
answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

print("Question:", query)
print("Answer:", answer)