import json
from collections import Counter

fields = [
    "id",
    "title",
    "keywords",
    "abstract",
    "introduction",
    "results",
    "discussion",
    "conclusion",
]

missing_counts = Counter()
total_docs = 0

with open("../small_data/removed.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        total_docs += 1

        if not doc.get("id", "").strip():
            missing_counts["id"] += 1
        if not doc.get("title", "").strip():
            missing_counts["title"] += 1

        kws = doc.get("keywords")
        if not isinstance(kws, list) or len(kws) == 0:
            missing_counts["keywords"] += 1

        for sec in ["abstract", "introduction", "results", "discussion", "conclusion"]:
            if not doc.get(sec, "").strip():
                missing_counts[sec] += 1

print(f"Total documents scanned: {total_docs}")
for field in fields:
    print(f"{field}: {missing_counts[field]} is missing")
