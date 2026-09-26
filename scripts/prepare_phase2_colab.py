#!/usr/bin/env python3
"""Prepare a Phase 2 train JSONL on Colab without local gitignored data/.

- Spatial rows: generated here (current synth = bare labels + D4 where wired).
- General rows: downloaded from `dwidlee/systemone-lite-general`.
- Chess rows: local `data/chess_train_5k_2d.jsonl` if present, else filtered from
  `dwidlee/systemone-lite-phase2` (gym == chess).
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from collections import Counter
from pathlib import Path

from systemone_lite.synth.connect4 import generate_connect4_samples
from systemone_lite.synth.game2048 import generate_2048_samples
from systemone_lite.synth.gridworld import generate_gridworld_samples
from systemone_lite.synth.sokoban import generate_sokoban_samples

ROOT = Path(__file__).resolve().parents[1]

SPATIAL = {
    "sokoban": generate_sokoban_samples,
    "game2048": generate_2048_samples,
    "gridworld": generate_gridworld_samples,
    "connect4": generate_connect4_samples,
}

FORBIDDEN = (
    "RECOMMENDED",
    "OPTIMAL",
    "BLOCK THREAT",
    "BLOCKED",
    "CRITICAL:",
    "SAFE:",
    "Push box",
    "Blocked wall",
    "hazard_nearby",
    "immediate_opponent_threat",
    "safest and fastest",
    "Prioritize winning",
)


def _parse_jsonish(value: object) -> object:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text[:1] in "{[":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return value
    return value


def hf_row_to_jsonl(ex: dict) -> dict:
    keys = _parse_jsonish(ex.get("criteria_keys") or [])
    vals = _parse_jsonish(ex.get("criteria_values") or [])
    if not isinstance(keys, list) or not isinstance(vals, list):
        raise ValueError(f"bad criteria lists: {type(keys)} {type(vals)}")
    if len(keys) != len(vals):
        raise ValueError(f"criteria length mismatch {len(keys)} vs {len(vals)}")
    meta = _parse_jsonish(ex.get("meta") or {})
    if not isinstance(meta, dict):
        meta = {"raw_meta": meta}
    return {
        "task": str(ex["task"]),
        "state": _parse_jsonish(ex["state"]),
        "instructions": str(ex["instructions"]),
        "criteria": {str(k): str(v) for k, v in zip(keys, vals, strict=True)},
        "label_alias": str(ex["label_alias"]),
        "label_key": str(ex["label_key"]),
        "meta": meta,
    }


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


def upsample(rows: list[dict], target: int, rng: random.Random) -> list[dict]:
    if not rows:
        raise ValueError("cannot upsample empty list")
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


def ensure_gym(row: dict, default: str) -> dict:
    meta = dict(row.get("meta") or {})
    if not meta.get("gym"):
        meta["gym"] = default
    out = dict(row)
    out["meta"] = meta
    return out


def validate(rows: list[dict], name: str) -> None:
    gyms: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        blob = json.dumps(row, ensure_ascii=False)
        for kw in FORBIDDEN:
            if kw in blob:
                raise ValueError(f"{name} row {idx} contains forbidden {kw!r}")
        if row["label_alias"] not in row["criteria"]:
            raise ValueError(f"{name} row {idx} label_alias missing from criteria")
        gyms[str((row.get("meta") or {}).get("gym", "?"))] += 1
    print(f"{name}: {len(rows)} rows ok; gyms={dict(gyms)}")


def load_general_from_hf(seed: int) -> list[dict]:
    from datasets import load_dataset

    print("Loading dwidlee/systemone-lite-general train ...", flush=True)
    ds = load_dataset("dwidlee/systemone-lite-general", split="train")
    rows = [ensure_gym(hf_row_to_jsonl(ds[i]), "general") for i in range(len(ds))]
    print(f"  general rows: {len(rows)}", flush=True)
    return rows


def load_chess(limit: int | None, seed: int) -> list[dict]:
    for local in (
        ROOT / "data" / "chess_train_staged.jsonl",
        ROOT / "data" / "chess_train_5k_2d.jsonl",
    ):
        if local.exists():
            print(f"Loading chess from {local} ...", flush=True)
            rows = [ensure_gym(r, "chess") for r in load_jsonl(local)]
            if local.name.endswith("_staged.jsonl"):
                rows = [r for r in rows if r.get("task") != "move"]
            return rows

    from datasets import load_dataset

    print("Loading chess rows from dwidlee/systemone-lite-phase2 ...", flush=True)
    ds = load_dataset("dwidlee/systemone-lite-phase2", split="train")
    rows: list[dict] = []
    for i in range(len(ds)):
        ex = ds[i]
        meta = _parse_jsonish(ex.get("meta") or {})
        gym = str(meta.get("gym", "")) if isinstance(meta, dict) else ""
        if gym != "chess" and not str(ex.get("task", "")).startswith("chess"):
            continue
        rows.append(ensure_gym(hf_row_to_jsonl(ex), "chess"))
        if limit is not None and len(rows) >= max(limit * 2, 5000):
            break
    print(f"  chess rows: {len(rows)}", flush=True)
    if not rows:
        raise RuntimeError("no chess rows found locally or on HF phase2")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab-friendly Phase 2 JSONL builder")
    parser.add_argument("--spatial-per-game", type=int, default=4000)
    parser.add_argument("--chess-target", type=int, default=4000)
    parser.add_argument("--general-target", type=int, default=4000)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "phase2_train_colab.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    spatial: list[dict] = []
    for idx, (name, gen) in enumerate(SPATIAL.items()):
        print(f"Generating {name} x{args.spatial_per_game} ...", flush=True)
        samples = gen(args.spatial_per_game, seed=args.seed + idx * 1000)
        spatial.extend(ensure_gym(s.to_json(), name) for s in samples)

    general = upsample(load_general_from_hf(args.seed), args.general_target, rng)
    chess = upsample(load_chess(None, args.seed), args.chess_target, rng)

    rows = spatial + general + chess
    rng.shuffle(rows)
    validate(rows, "phase2_colab")
    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
