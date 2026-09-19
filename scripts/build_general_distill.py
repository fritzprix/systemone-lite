#!/usr/bin/env python3
"""Build a general-purpose System One distillation JSONL."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from systemone_lite.synth import (
    generate_allocator_episode,
    generate_debate_episode,
    generate_ticket_episode,
)

ROOT = Path(__file__).resolve().parents[1]

GYMS = {
    "ticket": generate_ticket_episode,
    "alloc": generate_allocator_episode,
    "debate": generate_debate_episode,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="General System One synthetic distill")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "general_distill.jsonl",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2000,
        help="Episodes per gym (each episode yields multiple typed questions)",
    )
    parser.add_argument(
        "--gyms",
        default="ticket,alloc,debate",
        help="Comma-separated gyms: ticket,alloc,debate",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    selected = [g.strip() for g in args.gyms.split(",") if g.strip()]
    for name in selected:
        if name not in GYMS:
            raise SystemExit(f"unknown gym {name!r}; choose from {sorted(GYMS)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    task_counts: Counter[str] = Counter()
    gym_counts: Counter[str] = Counter()
    n = 0

    with args.out.open("w", encoding="utf-8") as fh:
        for gym_name in selected:
            gen = GYMS[gym_name]
            for _ in range(args.episodes):
                for sample in gen(rng):
                    fh.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")
                    task_counts[sample.task] += 1
                    gym_counts[sample.meta.get("gym", gym_name)] += 1
                    n += 1

    print(f"wrote {n} samples → {args.out}")
    print("by gym:", dict(gym_counts))
    print("by task:", dict(task_counts))


if __name__ == "__main__":
    main()
