import json
import argparse
from collections import Counter

def has_nonempty_section(doc):
    """
    Returns True if the given document dict has at least one non-empty value
    among the sections.
    """
    sections = ["introduction", "results", "discussion", "conclusion"]
    for sec in sections:
        text = doc.get(sec, "")
        if isinstance(text, str) and text.strip():
            return True
    return False

def filter_jsonl(input_path, output_path):
    """
    Read through input_path and write only those documents
    that have at least one non-empty section to output_path.
    """
    kept_count = 0
    removed_count = 0
    section_counts = Counter()

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line_num, line in enumerate(infile, start=1):
            line = line.rstrip("\n")
            if not line:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                removed_count += 1
                continue

            count = sum(
                1
                for s in ["introduction", "results", "discussion", "conclusion"]
                if isinstance(doc.get(s, ""), str) and doc.get(s, "").strip()
            )
            section_counts[count] += 1

            if count > 0:
                json_line = json.dumps(doc, ensure_ascii=False)
                outfile.write(json_line + "\n")
                kept_count += 1
            else:
                removed_count += 1

    print(f"Total documents processed: {kept_count + removed_count:,}")
    print(f"Documents kept (>=1 section): {kept_count:,}")
    print(f"Documents removed (0 sections): {removed_count:,}")
    print("Section‐count distribution (number of non-empty sections → document count):")
    for num_secs, freq in sorted(section_counts.items()):
        print(f"  {num_secs:>2} → {freq:,}")

def main():
    parser = argparse.ArgumentParser(
        description="Filter out the document that has 0 non‐empty sections."
    )
    parser.add_argument(
        "input", help="Path to the input data"
    )
    parser.add_argument(
        "output", help="Path to the output data (filtered, only docs with ≥1 section)"
    )
    args = parser.parse_args()

    filter_jsonl(args.input, args.output)

if __name__ == "__main__":
    main()
