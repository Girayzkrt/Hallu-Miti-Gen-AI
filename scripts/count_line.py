import argparse
import sys

def count_lines(filename):
    """
    Counts the number of lines in the given file.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error while reading '{filename}': {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Count the number of lines in a JSONL file."
    )
    parser.add_argument(
        'file',
        metavar='JSONL_FILE',
    )
    args = parser.parse_args()

    count = count_lines(args.file)
    if count is not None:
        print(f"Total lines in '{args.file}': {count}")

if __name__ == "__main__":
    main()
