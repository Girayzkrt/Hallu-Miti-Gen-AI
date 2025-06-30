import json
import re
from pathlib import Path

# Regular expressions
RX_PAREN_CITATION = re.compile(r'\(\s*(?:\d+(?:\s*,\s*\d+)*|[A-Z][A-Za-z\s\-\.]+ et al\.,? \d{4}[a-z]?|[A-Z][A-Za-z\s\-\.]+,? \d{4}[a-z]?)\s*\)')
RX_BRACK_CITATION = re.compile(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]')
RX_AUTHOR_YEAR = re.compile(r'\([A-Za-z][^)]*\d{4}[^)]*\)')
RX_LATEX_ENV      = re.compile(r'\\(usepackage|begin|end)\s*{[^}]*}')
RX_LATEX_CMD_ARG  = re.compile(r'\\[a-zA-Z]+\s*{[^}]*}')
RX_LATEX_CMD      = re.compile(r'\\[a-zA-Z]+')
RX_MATH_DOLLAR    = re.compile(r'\${1,2}[^$]*\${1,2}')
RX_FIG_TABLE = re.compile(
    r'\b(Fig\.|Figure|Table|Supplementary\s+Fig\.)\s*\d+[A-Za-z]*\b',
    flags=re.IGNORECASE
)
RX_FIG_TABLE_PL = re.compile(
    r'\b(figs?\.|tables?)\s+[A-Za-z]*\d+(?:\s*(?:and|,)?\s*[A-Za-z]*\d+)*',
    flags=re.IGNORECASE
)
RX_ENTRY_N = re.compile(r'\(entry\s*\d+\)', flags=re.IGNORECASE)
RX_WHITESPACE = re.compile(r'\s+')
RX_SPACE_PUNCT = re.compile(r'\s+([.,;:!?])')


def clean_text(text: str) -> str:
    """
    Strips citations, LaTeX, dimensions, figure/table refs
    """
    text = RX_PAREN_CITATION.sub('', text)
    text = RX_BRACK_CITATION.sub('', text)
    text = RX_AUTHOR_YEAR.sub('', text)
    text = RX_LATEX_ENV.sub('', text)
    text = RX_LATEX_CMD_ARG.sub('', text)
    text = RX_LATEX_CMD.sub('', text)
    text = RX_MATH_DOLLAR.sub('', text)
    text = RX_FIG_TABLE.sub('', text)
    text = RX_FIG_TABLE_PL.sub('', text)
    text = RX_ENTRY_N.sub('', text)
    text = text.replace('\\\\', '')
    text = RX_WHITESPACE.sub(' ', text).strip()
    text = RX_SPACE_PUNCT.sub(r'\1', text)
    return text

def clean_regex(input_file: str | Path, output_file: str | Path) -> None:
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            for field in ('title', 'abstract', 'introduction',
                          'results', 'discussion', 'conclusion'):
                if field in data and isinstance(data[field], str):
                    data[field] = clean_text(data[field])

            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_path  = '../small_data/parsed_pmc_1_small.jsonl'
    output_path = '../small_data/parsed_pmc_1_small_cleaned.jsonl'
    clean_regex(input_path, output_path)
