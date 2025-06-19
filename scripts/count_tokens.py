from transformers import GPT2TokenizerFast
import json
import sys

def count_tokens_in_jsonl(file_path, model="gpt-4"):
    """
    Counts the number of tokens in each line of a JSONL file using the specified model's tokenizer.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(model)

    token_counts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            tokens = tokenizer.encode(line)
            token_count = len(tokens)

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                data = None

            token_counts.append((i, token_count, data))

    return token_counts


if __name__ == "__main__":
    file_path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt2"

    counts = count_tokens_in_jsonl(file_path, model)
    for line_number, token_count, _ in counts:
        print(f"Line {line_number}: {token_count} tokens")