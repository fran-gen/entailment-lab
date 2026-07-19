#!/usr/bin/env python3
"""Build a deterministic SNLI-derived smoke test dataset."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


SOURCE_PATH = Path("data/raw/snli/snli_1.0_train.jsonl")
OUTPUT_DIR = Path("data/hf/snli-smoke-test")
OUTPUT_PATH = OUTPUT_DIR / "train.jsonl"
LOCAL_OUTPUT_DIR = Path("data/custom")
LOCAL_OUTPUT_PATH = LOCAL_OUTPUT_DIR / "snli_smoke_test.jsonl"

VALID_LABELS = ("entailment", "contradiction", "neutral")
TRIPLET_GROUPS = 33
TOTAL_EXAMPLES = 100
LABEL_TARGETS = {
    "entailment": 34,
    "contradiction": 33,
    "neutral": 33,
}


def load_snli_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as file:
        for line in file:
            row = json.loads(line)
            label = row.get("gold_label")
            annotator_labels = row.get("annotator_labels", [])
            if label not in VALID_LABELS:
                continue
            if len(annotator_labels) != 1 or annotator_labels[0] != label:
                continue
            rows.append(row)
    return rows


def build_triplet_rows(rows: list[dict]) -> list[dict]:
    by_caption_id: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_caption_id[row["captionID"]][row["gold_label"]] = row

    selected: list[dict] = []
    used_pair_ids: set[str] = set()

    for caption_id in sorted(by_caption_id):
        label_map = by_caption_id[caption_id]
        if not all(label in label_map for label in VALID_LABELS):
            continue
        for label in VALID_LABELS:
            row = label_map[label]
            selected.append(row)
            used_pair_ids.add(row["pairID"])
        if len(selected) >= TRIPLET_GROUPS * len(VALID_LABELS):
            break

    if len(selected) != TRIPLET_GROUPS * len(VALID_LABELS):
        raise ValueError("Unable to build the requested number of caption triplets.")

    for row in rows:
        if row["gold_label"] != "entailment":
            continue
        if row["pairID"] in used_pair_ids:
            continue
        selected.append(row)
        break

    if len(selected) != TOTAL_EXAMPLES:
        raise ValueError("Unable to add the final balancing example.")

    return selected


def infer_difficulty(label: str) -> str:
    if label == "entailment":
        return "easy"
    if label == "contradiction":
        return "easy"
    return "medium"


def infer_selection_strategy(index: int) -> str:
    if index <= TRIPLET_GROUPS * len(VALID_LABELS):
        return "caption_triplet"
    return "balancing_extra"


def to_internal_record(index: int, row: dict) -> dict:
    label = row["gold_label"]
    return {
        "id": f"snli_smoke_{index:03d}",
        "premise": row["sentence1"],
        "hypothesis": row["sentence2"],
        "label": label,
        "metadata": {
            "source": "snli",
            "source_split": "train",
            "source_pair_id": row["pairID"],
            "caption_id": row["captionID"],
            "phenomenon": "untyped_smoke_example",
            "reasoning_type": "generic_nli",
            "difficulty": infer_difficulty(label),
            "annotator_labels": row["annotator_labels"],
            "license": "cc-by-sa-4.0",
            "subset": "smoke_test",
            "selection_strategy": infer_selection_strategy(index),
        },
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    rows = load_snli_rows(SOURCE_PATH)
    selected_rows = build_triplet_rows(rows)
    records = [to_internal_record(index, row) for index, row in enumerate(selected_rows, start=1)]

    label_counts = defaultdict(int)
    for record in records:
        label_counts[record["label"]] += 1

    if dict(label_counts) != LABEL_TARGETS:
        raise ValueError(f"Unexpected label counts: {dict(label_counts)}")

    write_jsonl(OUTPUT_PATH, records)
    write_jsonl(LOCAL_OUTPUT_PATH, records)

    print(f"Wrote {len(records)} examples to {OUTPUT_PATH}")
    print(f"Wrote {len(records)} examples to {LOCAL_OUTPUT_PATH}")
    print(f"Label counts: {dict(label_counts)}")


if __name__ == "__main__":
    main()
