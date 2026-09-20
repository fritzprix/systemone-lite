#!/usr/bin/env python3
"""Upload Phase 2 204.8k dataset (General + Chess + Spatial 2D) to Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "dwidlee/systemone-lite-phase2"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    print(f"Loading {path}...", flush=True)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            criteria = row.get("criteria") or {}
            state_val = row["state"]
            state_str = json.dumps(state_val, ensure_ascii=False) if isinstance(state_val, (dict, list)) else str(state_val)
            meta_val = row.get("meta") or {}
            meta_str = json.dumps(meta_val, ensure_ascii=False) if isinstance(meta_val, (dict, list)) else str(meta_val)

            rows.append(
                {
                    "task": str(row["task"]),
                    "state": state_str,
                    "instructions": str(row["instructions"]),
                    "criteria_keys": list(criteria.keys()),
                    "criteria_values": [str(v) for v in criteria.values()],
                    "label_alias": str(row["label_alias"]),
                    "label_key": str(row["label_key"]),
                    "meta": meta_str,
                }
            )
    print(f"  Loaded {len(rows)} rows from {path.name}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Phase 2 Dataset to Hugging Face")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "phase2_train_200k.jsonl")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "phase2_eval_4k.jsonl")
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

    train_rows = load_jsonl(args.train)
    test_rows = load_jsonl(args.test)

    print("Converting to Hugging Face DatasetDict...", flush=True)
    splits = {
        "train": Dataset.from_list(train_rows, features=features),
        "test": Dataset.from_list(test_rows, features=features),
    }
    ds = DatasetDict(splits)

    print(f"Pushing dataset to Hugging Face Hub: https://huggingface.co/datasets/{args.repo} ...", flush=True)
    ds.push_to_hub(args.repo, private=args.private)
    print(f"\n🎉 Successfully pushed to Hugging Face: https://huggingface.co/datasets/{args.repo}")
    print(f"Splits: { {k: len(v) for k, v in ds.items()} }")


if __name__ == "__main__":
    main()
