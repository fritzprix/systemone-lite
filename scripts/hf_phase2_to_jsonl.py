#!/usr/bin/env python3
"""Download `dwidlee/systemone-lite-phase2` (or local JSONL) and write train JSONL for chess_finetune.

HF rows store state/meta as JSON strings and criteria as parallel key/value lists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse_jsonish(value: object) -> object:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "{[":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                import ast

                try:
                    return ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    return value
        return value
    return value


def row_from_hf(ex: dict) -> dict:
    keys = _parse_jsonish(ex.get("criteria_keys") or [])
    vals = _parse_jsonish(ex.get("criteria_values") or [])
    if not isinstance(keys, list) or not isinstance(vals, list):
        raise ValueError(f"bad criteria lists: {type(keys)} {type(vals)}")
    if len(keys) != len(vals):
        raise ValueError(f"criteria_keys/values length mismatch: {len(keys)} vs {len(vals)}")
    criteria = {str(k): str(v) for k, v in zip(keys, vals, strict=True)}
    meta = _parse_jsonish(ex.get("meta") or {})
    if not isinstance(meta, dict):
        meta = {"raw_meta": meta}
    return {
        "task": str(ex["task"]),
        "state": _parse_jsonish(ex["state"]),
        "instructions": str(ex["instructions"]),
        "criteria": criteria,
        "label_alias": str(ex["label_alias"]),
        "label_key": str(ex["label_key"]),
        "meta": meta,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Phase 2 HF dataset → JSONL")
    parser.add_argument("--repo", default="dwidlee/systemone-lite-phase2")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "phase2_train.jsonl")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row cap (e.g. 20000 for Colab T4 smoke runs)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.repo} split={args.split} ...", flush=True)
    ds = load_dataset(args.repo, split=args.split)
    n = len(ds)
    print(f"  remote rows: {n}", flush=True)

    if args.limit is not None and args.limit < n:
        ds = ds.shuffle(seed=args.seed).select(range(args.limit))
        print(f"  capped to {len(ds)} rows (seed={args.seed})", flush=True)

    rows = [row_from_hf(ds[i]) for i in range(len(ds))]
    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
