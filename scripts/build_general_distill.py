#!/usr/bin/env python3
"""Build a properly mixed general System One distillation JSONL.

- Round-robin gym episodes (not gym-block sequential dumps)
- Stratified train / iid-eval split by task
- Separate hard eval with layout / paraphrase / option-subset shift
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from systemone_lite.chess_data import DistillSample
from systemone_lite.synth import (
    generate_allocator_episode,
    generate_debate_episode,
    generate_ticket_episode,
)

ROOT = Path(__file__).resolve().parents[1]

GYMS: dict[str, Callable[..., list[DistillSample]]] = {
    "ticket": generate_ticket_episode,
    "alloc": generate_allocator_episode,
    "debate": generate_debate_episode,
}


def write_jsonl(path: Path, rows: list[DistillSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample in rows:
            fh.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")


def stratified_split(
    rows: list[DistillSample],
    *,
    eval_frac: float,
    seed: int,
) -> tuple[list[DistillSample], list[DistillSample]]:
    by_task: dict[str, list[DistillSample]] = defaultdict(list)
    for row in rows:
        by_task[row.task].append(row)

    rng = random.Random(seed)
    train: list[DistillSample] = []
    eval_rows: list[DistillSample] = []
    for task, group in sorted(by_task.items()):
        rng.shuffle(group)
        n_eval = max(1, int(round(len(group) * eval_frac)))
        n_eval = min(n_eval, len(group) - 1) if len(group) > 1 else 0
        eval_rows.extend(group[:n_eval])
        train.extend(group[n_eval:])

    rng.shuffle(train)
    rng.shuffle(eval_rows)
    return train, eval_rows


def summarize(label: str, rows: list[DistillSample]) -> None:
    gyms = Counter(r.meta.get("gym", "?") for r in rows)
    tasks = Counter(r.task for r in rows)
    print(f"{label}: n={len(rows)} gyms={dict(gyms)}")
    print(f"  tasks={dict(tasks)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mixed general System One distill")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--episodes",
        type=int,
        default=4000,
        help="Episodes per gym for the train pool (before split)",
    )
    parser.add_argument(
        "--hard-episodes",
        type=int,
        default=600,
        help="Episodes per gym for hard held-out eval",
    )
    parser.add_argument("--eval-frac", type=float, default=0.1)
    parser.add_argument(
        "--gyms",
        default="ticket,alloc,debate",
        help="Comma-separated gyms: ticket,alloc,debate",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    selected = [g.strip() for g in args.gyms.split(",") if g.strip()]
    for name in selected:
        if name not in GYMS:
            raise SystemExit(f"unknown gym {name!r}; choose from {sorted(GYMS)}")

    train_rng = random.Random(args.seed)
    pool: list[DistillSample] = []
    # Round-robin episodes so the raw dump is already mixed.
    for i in range(args.episodes):
        for gym_name in selected:
            gen = GYMS[gym_name]
            # Per-episode child RNG keeps gyms independent but reproducible.
            ep_rng = random.Random(train_rng.randint(0, 2**31 - 1))
            pool.extend(gen(ep_rng, hard=False))

    train_rows, iid_eval = stratified_split(
        pool, eval_frac=args.eval_frac, seed=args.seed + 1
    )

    hard_rng = random.Random(args.seed + 10_000)
    hard_eval: list[DistillSample] = []
    for i in range(args.hard_episodes):
        for gym_name in selected:
            gen = GYMS[gym_name]
            ep_rng = random.Random(hard_rng.randint(0, 2**31 - 1))
            hard_eval.extend(gen(ep_rng, hard=True))

    out = args.out_dir
    write_jsonl(out / "general_train.jsonl", train_rows)
    write_jsonl(out / "general_eval.jsonl", iid_eval)
    write_jsonl(out / "general_eval_hard.jsonl", hard_eval)
    write_jsonl(out / "general_distill.jsonl", train_rows + iid_eval)

    summarize("train", train_rows)
    summarize("eval_iid", iid_eval)
    summarize("eval_hard", hard_eval)
    meta = {
        "seed": args.seed,
        "episodes_per_gym": args.episodes,
        "hard_episodes_per_gym": args.hard_episodes,
        "eval_frac": args.eval_frac,
        "gyms": selected,
        "n_train": len(train_rows),
        "n_eval_iid": len(iid_eval),
        "n_eval_hard": len(hard_eval),
    }
    (out / "general_build_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"meta → {out / 'general_build_meta.json'}")


if __name__ == "__main__":
    main()
