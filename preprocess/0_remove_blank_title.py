import json
import sys
import os
import tempfile

def cleanup_jsonl(input_path, temp_path):
    """
    Reads the JSONL file at input_path line-by-line, writes back only those lines
    whose "title" field is non-empty to temp_path. Returns a tuple (total_lines, deleted_lines).
    """
    total = 0
    deleted = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(temp_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = record.get("title", None)
            if title == "":
                deleted += 1
                continue

            outfile.write(line)

    return total, deleted

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <input.jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"Error: '{input_path}' does not exist or is not a file.")
        sys.exit(1)

    dirpath = os.path.dirname(os.path.abspath(input_path))
    fd, temp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp")
    os.close(fd)

    total_lines, deleted_lines = cleanup_jsonl(input_path, temp_path)

    os.replace(temp_path, input_path)

    print(f"Processed {total_lines} lines. Deleted {deleted_lines} lines.")

if __name__ == "__main__":
    main()
