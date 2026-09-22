#!/usr/bin/env python3
"""Audit train∩eval state contamination per gym.

Exit 1 if any gym's state_task overlap exceeds --max-overlap (default 0.0).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from systemone_lite.synth.leakage import fingerprint

ROOT = Path(__file__).resolve().parents[1]


def load_by_gym(path: Path) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            gym = str((row.get("meta") or {}).get("gym") or "?")
            by[gym].append(row)
    return by


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=ROOT / "data" / "phase2_train_200k.jsonl",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=ROOT / "data" / "phase2_eval_4k.jsonl",
    )
    parser.add_argument("--max-overlap", type=float, default=0.0)
    args = parser.parse_args()

    train = load_by_gym(args.train)
    eval_rows = load_by_gym(args.eval)

    print(f"train={args.train}")
    print(f"eval ={args.eval}")
    fails: list[str] = []
    total_eval = 0
    total_hits_st = 0
    total_hits_full = 0

    for gym in sorted(set(train) | set(eval_rows)):
        ev = eval_rows.get(gym, [])
        if not ev:
            continue
        tr = train.get(gym, [])
        t_state = {fingerprint(r, mode="state_only") for r in tr}
        t_st = {fingerprint(r, mode="state_task") for r in tr}
        t_full = {fingerprint(r, mode="full") for r in tr}
        hits_state = sum(1 for r in ev if fingerprint(r, mode="state_only") in t_state)
        hits_st = sum(1 for r in ev if fingerprint(r, mode="state_task") in t_st)
        hits_full = sum(1 for r in ev if fingerprint(r, mode="full") in t_full)
        n = len(ev)
        total_eval += n
        total_hits_st += hits_st
        total_hits_full += hits_full
        rate = hits_st / n
        print(
            f"{gym:20s} n_eval={n:4d}  state={hits_state/n:6.1%}  "
            f"state_task={rate:6.1%}  full={hits_full/n:6.1%}"
        )
        if rate > args.max_overlap + 1e-12:
            fails.append(f"{gym}: state_task overlap {rate:.1%} > {args.max_overlap}")

    if total_eval:
        print(
            f"\nTOTAL state_task={total_hits_st}/{total_eval} "
            f"({100 * total_hits_st / total_eval:.2f}%)  "
            f"full={total_hits_full}/{total_eval} "
            f"({100 * total_hits_full / total_eval:.2f}%)"
        )

    if fails:
        print("\nFAIL:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("\nPASS: all gyms within max-overlap")


if __name__ == "__main__":
    main()
