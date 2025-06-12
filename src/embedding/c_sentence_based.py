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

input_path = "data/parsed_pmc_2_s.jsonl"
output_path = "data/parsed_pmc_2_s_chunked.jsonl"

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=True))

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:  #encoding="utf-8" in Windows - gotta remove it when running on mac
    last_chunk_text = None
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
                tokenized = tokenizer.encode(chunk_text, add_special_tokens=True)
                if len(tokenized) > 512:
                    print(f"Warning: Chunk {chunk_id} in doc {doc_idx} has {len(tokenized)} tokens, truncating.")
                    tokenized = tokenized[:512]
                    chunk_text = tokenizer.decode(tokenized, skip_special_tokens=True)
                if chunk_text == last_chunk_text:
                    print(f"Skipping duplicate chunk {chunk_id} in doc {doc_idx}")
                    break
                last_chunk_text = chunk_text
                out = {
                    "original_index": doc_idx,
                    "section": field,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                }
                fout.write(json.dumps(out) + "\n")
                chunk_id += 1
        print(f"Processed document {doc_idx + 1}")
        print(f"Total chunks created: {chunk_id}")

"""
Processed document 1477991
"""