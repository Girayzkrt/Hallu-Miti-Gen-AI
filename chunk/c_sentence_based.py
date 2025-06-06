import json
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
"""	
Sentence-based chunking with token limit and sentence overlap
- Splits text into sentences with NLTK
- Builds chunks by sentences one by one to a chunnk until the token limit is reached (480 tokens)
- For each chunk after the first, it includes the last N sentences from the previous chunk to maintain context


-Cons?????????
- Max tokens: 480 ist genug oder nix


"""
#MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "intfloat/e5-large-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_TOKENS = 480
OVERLAP_SENTENCES = 2

input_path = "chunk_test.jsonl"
output_path = "chunked_output.jsonl"

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

with open(input_path, "r") as fin, open(output_path, "w") as fout:
    for doc_idx, line in enumerate(fin):
        item = json.loads(line)
        for field in item:
            if not item[field] or not isinstance(item[field], str):
                continue
            sentences = sent_tokenize(item[field])
            i = 0
            chunk_id = 0
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
                out = {
                    "original_index": doc_idx,
                    "section": field,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                }
                fout.write(json.dumps(out) + "\n")
                chunk_id += 1
