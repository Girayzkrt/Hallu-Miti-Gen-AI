import json
import re

def clean_text(text):
    """
    This function takes a text string and removes various unwanted elements—such as numeric and author-year citations in parentheses,
    IEEE-style bracketed citations, LaTeX commands and formatting (including bracketed options, inline commands,
    and standalone commands), dimension specifications (like -1.0in or 2.5cm), stray backslashes, non-ASCII characters,
    figure/table references, and other parenthetical entries—then returns the cleaned ASCII-only string.
    """
    text = re.sub(r'\(\s*\d+(?:\s*(?:,|\u2013|\u2014|-)\s*\d+)*\s*\)', '', text)
    text = re.sub(r'\[\s*\d+(?:\s*(?:,|\u2013|\u2014|-)\s*\d+)*\s*\]', '', text)
    text = re.sub(r'\([A-Za-z][^)]*\d{4}[^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]\{[^}]*\}', '', text)
    text = re.sub(r'-?\d+(?:\.\d+)?(?:in|cm|mm|pt)', '', text)
    text = re.sub(r'\\\\+', '', text)
    text = re.sub(r'\\(usepackage|begin|end)\s*{[^}]*}', '', text)
    text = re.sub(r'\\[a-zA-Z]+\s*{[^}]*}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
    text = text.replace('\n', ' ')
    text = text.encode('ascii', errors='ignore').decode()
    text = re.sub(
        r'\b(Fig\.|Figure|Table|Supplementary\s+Fig\.)\s*\d+[A-Za-z]*\b',
        '',
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r'\(entry\s*\d+\)', '', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\b(figs?\.|tables?)\s+[A-Za-z]*\d+(?:\s*(and|,)?\s*[A-Za-z]*\d+)*',
        '',
        text,
        flags=re.IGNORECASE
    )

    return text

def clean_citations_in_jsonl(input_file, output_file):
    print(f"Reading from: {input_file}")
    try:
        line_count = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
        print(f"Total lines to process: {line_count}")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile, start=1):
            if i % 1000 == 0:
                print(f"Processed {i} lines...", flush=True)
            try:
                data = json.loads(line)
                print(f"Line {i}: JSON loaded, starting cleaning...", flush=True)
                try:
                    for field in ['title', 'abstract', 'introduction', 'results', 'discussion', 'conclusion']:
                        if field in data and isinstance(data[field], str):
                            field_text = data[field]
                            text_length = len(field_text)
                            print(f"Line {i}: cleaning field '{field}' (length {text_length})", flush=True)
                            data[field] = clean_text(field_text)
                            print(f"Line {i}: finished cleaning field '{field}'", flush=True)
                except Exception as e:
                    print(f"Error cleaning line {i}: {e}", flush=True)
                print(f"Line {i}: writing cleaned JSON to output", flush=True)
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                print(f"Skipping line {i} due to JSON error: {e}")

input_path = '../small_data/parsed_pmc_1_small.jsonl'
output_path = '../small_data/preprocess.jsonl'
clean_citations_in_jsonl(input_path, output_path)