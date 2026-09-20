#!/usr/bin/env python3
"""Build balanced Phase 2 dataset: General NLP + 2D Chess + 2D Spatial Games.

Mixes three 10,800-sample balanced pillars:
  1. General System 1 NLP (10,800 rows: support, routing, sentiment, triage)
  2. 2D Chess Tactical Policy (10,800 rows: debiased board_2d_map)
  3. 2D Spatial Games (10,800 rows: 2,700 each for Sokoban, 2048, GridWorld, Connect4)
Total: 32,400 stratified samples.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
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
    parser = argparse.ArgumentParser(description="Build Phase 2 3-Way Stratified Mixed Dataset")
    parser.add_argument("--general", type=Path, default=ROOT / "data" / "general_train.jsonl")
    parser.add_argument("--chess", type=Path, default=ROOT / "data" / "chess_train_5k_2d.jsonl")
    parser.add_argument("--spatial", type=Path, default=ROOT / "data" / "spatial_train_10k.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "phase2_train.jsonl")
    parser.add_argument("--target-per-pillar", type=int, default=10800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading data pillars...")
    general_raw = [ensure_gym(r, str((r.get("meta") or {}).get("gym") or "general")) for r in load_jsonl(args.general)]
    chess_raw = [ensure_gym(r, "chess") for r in load_jsonl(args.chess)]
    spatial_raw = [ensure_gym(r, str((r.get("meta") or {}).get("gym") or "spatial")) for r in load_jsonl(args.spatial)]

    print(f"  Raw counts: General={len(general_raw)}, Chess={len(chess_raw)}, Spatial={len(spatial_raw)}")

    general_samples = upsample(general_raw, args.target_per_pillar, rng)
    chess_samples = upsample(chess_raw, args.target_per_pillar, rng)
    spatial_samples = upsample(spatial_raw, args.target_per_pillar, rng)

    phase2_all = general_samples + chess_samples + spatial_samples
    rng.shuffle(phase2_all)

    write_jsonl(args.out, phase2_all)

    gym_counts = Counter((r.get("meta") or {}).get("gym", "?") for r in phase2_all)
    print(f"\nSuccessfully wrote {len(phase2_all)} rows → {args.out}")
    print("Breakdown by gym:")
    for gym, cnt in sorted(gym_counts.items(), key=lambda x: -x[1]):
        print(f"  - {gym:<15}: {cnt:>6} samples ({cnt / len(phase2_all) * 100:.1f}%)")


if __name__ == "__main__":
    main()
