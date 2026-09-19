#!/usr/bin/env python3
"""Upload general System One distill JSONL to Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "dwidlee/systemone-lite-general"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        criteria = row.get("criteria") or {}
        rows.append(
            {
                "task": row["task"],
                "state": json.dumps(row["state"], ensure_ascii=False),
                "instructions": row["instructions"],
                "criteria_keys": list(criteria.keys()),
                "criteria_values": list(criteria.values()),
                "label_alias": row["label_alias"],
                "label_key": row["label_key"],
                "meta": json.dumps(row.get("meta") or {}, ensure_ascii=False),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "general_train.jsonl")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "general_eval.jsonl")
    parser.add_argument(
        "--test-hard",
        type=Path,
        default=ROOT / "data" / "general_eval_hard.jsonl",
    )
    parser.add_argument("--all", type=Path, default=ROOT / "data" / "general_distill.jsonl")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    features = Features(
        {
            "task": Value("string"),
            "state": Value("string"),
            "instructions": Value("string"),
            "criteria_keys": Sequence(Value("string")),
            "criteria_values": Sequence(Value("string")),
            "label_alias": Value("string"),
            "label_key": Value("string"),
            "meta": Value("string"),
        }
    )
    splits = {
        "train": Dataset.from_list(load_jsonl(args.train), features=features),
        "test": Dataset.from_list(load_jsonl(args.test), features=features),
        "test_hard": Dataset.from_list(load_jsonl(args.test_hard), features=features),
        # HF reserves the name "all"; use "full" for train+iid pool.
        "full": Dataset.from_list(load_jsonl(args.all), features=features),
    }
    ds = DatasetDict(splits)
    ds.push_to_hub(args.repo, private=args.private)
    print(f"pushed {args.repo}: { {k: len(v) for k, v in ds.items()} }")


if __name__ == "__main__":
    main()
