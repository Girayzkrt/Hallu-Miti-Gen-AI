import json
import sys
import os
import tempfile

def clean_data(input_path, temp_path):
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
            abstract = record.get("abstract", None)
            if abstract == "":
                deleted += 1
                continue
            outfile.write(line)

    return total, deleted

def main():
    input_path = sys.argv[1]
    dirpath = os.path.dirname(os.path.abspath(input_path))

    fd, temp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp")
    os.close(fd)

    total_lines, deleted_lines = clean_data(input_path, temp_path)

    os.replace(temp_path, input_path)

    print(f"Processed {total_lines} lines. Deleted {deleted_lines} lines.")

if __name__ == "__main__":
    main()
