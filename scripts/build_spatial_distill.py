#!/usr/bin/env python3
"""Build synthetic spatial 2D text map distillation JSONL for System One SFT.

Supports all 4 spatial 2D game domains:
  1. Sokoban (box-pushing & deadlock avoidance)
  2. 2048 (4x4 tile merging & overflow prevention)
  3. GridWorld (procedural maze navigation & hazard avoidance)
  4. Connect Four (7-column drop & 4-in-a-row threat defense)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from systemone_lite.synth.connect4 import generate_connect4_samples
from systemone_lite.synth.game2048 import generate_2048_samples
from systemone_lite.synth.gridworld import generate_gridworld_samples
from systemone_lite.synth.sokoban import generate_sokoban_samples

ROOT = Path(__file__).resolve().parents[1]

GENERATORS = {
    "sokoban": generate_sokoban_samples,
    "game2048": generate_2048_samples,
    "gridworld": generate_gridworld_samples,
    "connect4": generate_connect4_samples,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2D spatial text-map distillation datasets")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "spatial_distill.jsonl")
    parser.add_argument(
        "--games",
        type=str,
        default="sokoban,game2048,gridworld,connect4",
        help="Comma-separated list of games (sokoban, game2048, gridworld, connect4)",
    )
    parser.add_argument("--samples-per-game", type=int, default=500, help="Number of samples per game")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard", action="store_true", help="Include perturbed/hard representations")
    args = parser.parse_args()

    selected_games = [g.strip() for g in args.games.split(",") if g.strip()]
    for g in selected_games:
        if g not in GENERATORS:
            raise ValueError(f"Unknown game {g!r}. Available: {list(GENERATORS.keys())}")

    all_samples = []
    rng = random.Random(args.seed)

    print(f"Generating spatial datasets for: {selected_games}")
    for idx, game_name in enumerate(selected_games):
        gen_fn = GENERATORS[game_name]
        game_seed = args.seed + idx * 1000
        samples = gen_fn(args.samples_per_game, seed=game_seed, hard=args.hard)
        all_samples.extend(samples)
        print(f"  ✓ {game_name:<10}: {len(samples)} samples generated")

    rng.shuffle(all_samples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for s in all_samples:
            fh.write(json.dumps(s.to_json(), ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_samples)} total spatial samples → {args.out}")


if __name__ == "__main__":
    main()
