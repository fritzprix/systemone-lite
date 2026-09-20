#!/usr/bin/env python3
"""Merge general + chess JSONL into a stratified-ready mixed train set."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_gym(row: dict, default: str) -> dict:
    meta = dict(row.get("meta") or {})
    if not meta.get("gym"):
        meta["gym"] = default
    row = dict(row)
    row["meta"] = meta
    return row


def upsample(rows: list[dict], target: int, rng: random.Random) -> list[dict]:
    if len(rows) >= target:
        out = list(rows)
        rng.shuffle(out)
        return out[:target]
    out: list[dict] = []
    while len(out) < target:
        chunk = list(rows)
        rng.shuffle(chunk)
        out.extend(chunk)
    rng.shuffle(out)
    return out[:target]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--general",
        type=Path,
        default=ROOT / "data" / "general_train.jsonl",
    )
    parser.add_argument(
        "--chess",
        type=Path,
        default=ROOT / "data" / "chess_train_5k_2d.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "mixed_train.jsonl",
    )
    parser.add_argument(
        "--chess-target",
        type=int,
        default=10800,
        help="Upsample chess rows to this count (~one general gym)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    general = [ensure_gym(r, str((r.get("meta") or {}).get("gym") or "general")) for r in load_jsonl(args.general)]
    chess = [ensure_gym(r, "chess") for r in load_jsonl(args.chess)]
    chess_up = upsample(chess, args.chess_target, rng)

    mixed = general + chess_up
    rng.shuffle(mixed)
    write_jsonl(args.out, mixed)

    from collections import Counter

    gyms = Counter((r.get("meta") or {}).get("gym", "?") for r in mixed)
    print(f"wrote {len(mixed)} → {args.out}")
    print("by gym:", dict(gyms))
    print(f"chess raw={len(chess)} upsampled={len(chess_up)}")


if __name__ == "__main__":
    main()
