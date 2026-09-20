#!/usr/bin/env python3
"""Build large-scale Phase 2 dataset (~204,800 rows) for ~6-hour deep spatial training.

Dataset Composition:
  - 🎮 2D Spatial Games (140,000 rows, ~68.4%):
      * Sokoban: 35,000 rows
      * 2048: 35,000 rows
      * GridWorld: 35,000 rows
      * Connect Four: 35,000 rows
  - ♟️ 2D Chess Tactical Policy: 32,400 rows (~15.8%)
  - 📝 General System 1 NLP: 32,400 rows (~15.8%)
Total: 204,800 stratified samples (~51,200 training steps at batch size 4).

Also produces held-out evaluation set:
  - `data/phase2_eval_4k.jsonl`: 4,000 rows (2,000 spatial + 1,000 chess + 1,000 general).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from systemone_lite.synth.connect4 import generate_connect4_samples
from systemone_lite.synth.game2048 import generate_2048_samples
from systemone_lite.synth.gridworld import generate_gridworld_samples
from systemone_lite.synth.sokoban import generate_sokoban_samples

ROOT = Path(__file__).resolve().parents[1]

SPATIAL_GENERATORS = {
    "sokoban": generate_sokoban_samples,
    "game2048": generate_2048_samples,
    "gridworld": generate_gridworld_samples,
    "connect4": generate_connect4_samples,
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


def validate_dataset(rows: list[dict], name: str = "Dataset") -> None:
    """Rigorous schema and neutrality validation."""
    forbidden_keywords = ["RECOMMENDED:", "OPTIMAL:", "BLOCK THREAT:"]
    aliases_seen = Counter()
    gyms_seen = Counter()

    for idx, r in enumerate(rows):
        # 1. Required keys
        for key in ["task", "state", "instructions", "criteria", "label_alias", "label_key"]:
            if key not in r:
                raise ValueError(f"Row {idx} missing required key: {key}")

        criteria = r["criteria"]
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f"Row {idx} criteria must be a non-empty dict")

        # 2. Check for keyword hints / leaks
        text_dump = json.dumps(r, ensure_ascii=False)
        for kw in forbidden_keywords:
            if kw in text_dump:
                raise ValueError(f"Row {idx} contains forbidden heuristic keyword: {kw}")

        # 3. Label validity
        if r["label_alias"] not in criteria:
            raise ValueError(f"Row {idx} label_alias {r['label_alias']!r} not in criteria keys")

        aliases_seen[r["label_alias"]] += 1
        meta = r.get("meta") or {}
        gyms_seen[meta.get("gym", "unknown")] += 1

    print(f"\n✅ {name} Validation Passed ({len(rows)} samples):")
    print(f"   - Forbidden keywords check: 0 occurrences detected")
    print(f"   - Label alias distribution: {dict(aliases_seen.most_common(5))}")
    print(f"   - Gym distribution:")
    for gym, cnt in sorted(gyms_seen.items(), key=lambda x: -x[1]):
        print(f"       • {gym:<18}: {cnt:>7} samples ({cnt / len(rows) * 100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 204,800-sample Phase 2 Dataset")
    parser.add_argument("--spatial-per-game", type=int, default=35000, help="Samples per 2D game (default 35k)")
    parser.add_argument("--chess-target", type=int, default=32400, help="Chess samples (default 32.4k)")
    parser.add_argument("--general-target", type=int, default=32400, help="General NLP samples (default 32.4k)")
    parser.add_argument("--eval-spatial-per-game", type=int, default=500, help="Eval spatial samples per game")
    parser.add_argument("--train-out", type=Path, default=ROOT / "data" / "phase2_train_200k.jsonl")
    parser.add_argument("--eval-out", type=Path, default=ROOT / "data" / "phase2_eval_4k.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 1. Generate 2D Spatial Training Samples
    print(f"=== 1. Generating 2D Spatial Training Samples ({args.spatial_per_game} per game) ===")
    spatial_train_samples: list[dict] = []
    for idx, (game_name, gen_fn) in enumerate(SPATIAL_GENERATORS.items()):
        game_seed = args.seed + idx * 1000
        print(f"  Generating {game_name}...", flush=True)
        raw_samples = gen_fn(args.spatial_per_game, seed=game_seed)
        spatial_train_samples.extend([ensure_gym(s.to_json(), game_name) for s in raw_samples])
        print(f"    ✓ {game_name}: {len(raw_samples)} samples generated", flush=True)

    # 2. Generate 2D Spatial Eval Samples
    print(f"\n=== 2. Generating 2D Spatial Evaluation Samples ({args.eval_spatial_per_game} per game) ===")
    spatial_eval_samples: list[dict] = []
    for idx, (game_name, gen_fn) in enumerate(SPATIAL_GENERATORS.items()):
        game_seed = args.seed + 90000 + idx * 1000
        raw_eval = gen_fn(args.eval_spatial_per_game, seed=game_seed)
        spatial_eval_samples.extend([ensure_gym(s.to_json(), game_name) for s in raw_eval])

    # 3. Load & Scale General NLP + Chess
    print(f"\n=== 3. Loading General NLP & Chess 2D Datasets ===")
    general_raw = [ensure_gym(r, str((r.get("meta") or {}).get("gym") or "general")) for r in load_jsonl(ROOT / "data" / "general_train.jsonl")]
    chess_raw = [ensure_gym(r, "chess") for r in load_jsonl(ROOT / "data" / "chess_train_5k_2d.jsonl")]

    general_train = upsample(general_raw, args.general_target, rng)
    chess_train = upsample(chess_raw, args.chess_target, rng)

    # 4. Assemble Final Phase 2 Training Dataset
    print(f"\n=== 4. Assembling Final Phase 2 Train Dataset ===")
    phase2_train = spatial_train_samples + chess_train + general_train
    rng.shuffle(phase2_train)

    validate_dataset(phase2_train, name="Phase 2 Training Set")
    write_jsonl(args.train_out, phase2_train)
    print(f"Wrote {len(phase2_train)} samples → {args.train_out}")

    # 5. Assemble Eval Dataset
    print(f"\n=== 5. Assembling Final Phase 2 Eval Dataset ===")
    general_eval = [ensure_gym(r, str((r.get("meta") or {}).get("gym") or "general")) for r in load_jsonl(ROOT / "data" / "general_eval.jsonl")[:1000]]
    chess_eval = [ensure_gym(r, "chess") for r in load_jsonl(ROOT / "data" / "chess_eval_5k_2d.jsonl")[:1000]]
    phase2_eval = spatial_eval_samples + chess_eval + general_eval
    rng.shuffle(phase2_eval)

    validate_dataset(phase2_eval, name="Phase 2 Evaluation Set")
    write_jsonl(args.eval_out, phase2_eval)
    print(f"Wrote {len(phase2_eval)} samples → {args.eval_out}")


if __name__ == "__main__":
    main()
