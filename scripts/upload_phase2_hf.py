#!/usr/bin/env python3
"""Upload Phase 2 dataset (train + held-out test) to Hugging Face Hub.

Emphasizes **zero train∩test state contamination** after the 2026-09-22 scrub
(held-out vocab/topics + state-hash rejection + nlp_cloze).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "dwidlee/systemone-lite-phase2"

DATASET_CARD = """---
license: apache-2.0
pretty_name: systemone-lite Phase 2 distill
task_categories:
  - text-classification
  - multiple-choice
tags:
  - system-one
  - alias-ce
  - spatial
  - cloze
  - zero-leakage
size_categories:
  - 100K<n<1M
---

# systemone-lite-phase2

Typed System One distill rows (`task` / `state` / `instructions` / criteria /
`label_alias`) for [`systemone-lite`](https://github.com/fritzprix/systemone-lite).

## Critical: train / test hygiene (2026-09-22)

Earlier local mixes had **severe train∩eval state leakage** (debate ~91%,
word_games ~87%, connect4 ~37% state_task overlap). This Hub revision is rebuilt
with **0.00%** train∩test overlap on `state_task` fingerprints
(`scripts/audit_train_eval_overlap.py`).

| Mechanism | Detail |
|---|---|
| `word_games` | Disjoint TRAIN / EVAL vocabularies |
| `debate_judge` | Held-out topics (eval never sees train motions) |
| Spatial / chess / CA | Eval states rejected against train state hashes |
| `nlp_cloze` | WikiText-2 **document-disjoint** (train split → train, test split → eval) |

**Do not mix `train` rows into `test` (or vice versa) when building local evals.**

## Splits

| Split | Rows | Role |
|---|---:|---|
| `train` | {n_train} | SFT |
| `test` | {n_test} | Held-out eval only |

Gym counts (train): `{train_gyms}`

## Schema

Flat columns for Hub friendliness: `state` and `meta` are JSON strings;
`criteria_keys` / `criteria_values` parallel lists.

## Cite / notes

- Leakage + cloze write-up: repo `docs/NOTE_S1_CLOZE_AND_LEAKAGE_2026-09-22.md`
- Not affiliated with TypeSafe AI / JevBench official leaderboards.
"""


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
            state_str = (
                json.dumps(state_val, ensure_ascii=False)
                if isinstance(state_val, (dict, list))
                else str(state_val)
            )
            meta_val = row.get("meta") or {}
            meta_str = (
                json.dumps(meta_val, ensure_ascii=False)
                if isinstance(meta_val, (dict, list))
                else str(meta_val)
            )

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


def gym_counts(rows: list[dict]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        try:
            meta = json.loads(r["meta"]) if isinstance(r["meta"], str) else (r["meta"] or {})
        except json.JSONDecodeError:
            meta = {}
        c[str(meta.get("gym") or "?")] += 1
    return dict(sorted(c.items(), key=lambda x: (-x[1], x[0])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Phase 2 Dataset to Hugging Face")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "phase2_train_200k.jsonl")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "phase2_eval_4k.jsonl")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--skip-overlap-check",
        action="store_true",
        help="Skip local train∩test audit (not recommended)",
    )
    args = parser.parse_args()

    if not args.skip_overlap_check:
        print("=== Pre-upload overlap audit (must PASS) ===", flush=True)
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "audit_train_eval_overlap.py"),
                "--train",
                str(args.train),
                "--eval",
                str(args.test),
                "--max-overlap",
                "0.0",
            ],
            check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(
                "Refusing Hub upload: train∩test overlap audit failed. "
                "Rebuild with scripts/rebuild_heldout_eval.py first."
            )

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
    train_gyms = gym_counts(train_rows)

    print("Converting to Hugging Face DatasetDict...", flush=True)
    splits = {
        "train": Dataset.from_list(train_rows, features=features),
        "test": Dataset.from_list(test_rows, features=features),
    }
    ds = DatasetDict(splits)

    print(
        f"Pushing dataset to Hugging Face Hub: https://huggingface.co/datasets/{args.repo} ...",
        flush=True,
    )
    ds.push_to_hub(
        args.repo,
        private=args.private,
        commit_message=(
            "Zero-leakage rebuild: train/test disjoint (vocab/topic/state) + nlp_cloze"
        ),
    )

    card = DATASET_CARD.format(
        n_train=len(train_rows),
        n_test=len(test_rows),
        train_gyms=json.dumps(train_gyms),
    )
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Dataset card: document zero train/test leakage",
    )

    print(f"\nSuccessfully pushed: https://huggingface.co/datasets/{args.repo}")
    print(f"Splits: { {k: len(v) for k, v in ds.items()} }")
    print(f"Train gyms: {train_gyms}")


if __name__ == "__main__":
    main()
