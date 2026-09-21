#!/usr/bin/env python3
"""Audit Phase 2 JSONL for label collapse, leaks, and uniqueness.

Exit code 1 if any hard failure (single-class alert, coaching leak, empty-count leak).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = [
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
    "boxes_placed",
    "safest and fastest",
    "Prioritize winning",
    "DELIVERS CHECK",
    "CAPTURES enemy",
]

ALERT_TASKS = {
    "gridworld.hazard_alert",
    "sokoban.deadlock_alert",
    "game2048.overflow_alert",
    "connect4.threat_alert",
}


def row_fp(row: dict) -> str:
    blob = json.dumps(
        {
            "t": row["task"],
            "s": row["state"],
            "c": row["criteria"],
            "a": row["label_alias"],
            "k": row.get("label_key"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.md5(blob.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "phase2_train_200k.jsonl",
    )
    parser.add_argument(
        "--max-alert-collapse",
        type=float,
        default=0.65,
        help="Fail if any alert task majority class exceeds this fraction",
    )
    args = parser.parse_args()

    by_gym: dict[str, list[dict]] = defaultdict(list)
    with args.data.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            gym = str((row.get("meta") or {}).get("gym") or "?")
            by_gym[gym].append(row)

    hard_fails: list[str] = []
    warns: list[str] = []

    print(f"auditing {args.data} total={sum(len(v) for v in by_gym.values())}")

    for gym in sorted(by_gym):
        rows = by_gym[gym]
        uniq: set[str] = set()
        dups = 0
        dirty = 0
        empty_leak = 0
        for row in rows:
            h = row_fp(row)
            if h in uniq:
                dups += 1
            else:
                uniq.add(h)
            dump = json.dumps(row, ensure_ascii=False)
            if any(kw in dump for kw in FORBIDDEN):
                dirty += 1
            if "empty_cells_count" in (row.get("state") or {}):
                empty_leak += 1

        uniq_ratio = len(uniq) / max(len(rows), 1)
        print(
            f"\n== {gym} n={len(rows)} unique={len(uniq)} "
            f"({100 * uniq_ratio:.1f}%) dups={dups}"
        )
        if dirty:
            hard_fails.append(f"{gym}: {dirty} coaching-keyword rows")
        if empty_leak:
            hard_fails.append(
                f"{gym}: {empty_leak} rows leak empty_cells_count in state"
            )
        if gym in {"sokoban", "gridworld", "game2048", "connect4"} and uniq_ratio < 0.5:
            warns.append(f"{gym}: low unique ratio {uniq_ratio:.2f}")
        if gym == "cellular_automata":
            missing_legend = sum(
                1
                for r in rows
                if not (isinstance(r.get("state"), dict) and r["state"].get("legend"))
            )
            if missing_legend:
                hard_fails.append(
                    f"{gym}: {missing_legend} rows missing state.legend (rule text)"
                )
        if gym == "word_games" and uniq_ratio < 0.3:
            warns.append(f"{gym}: low unique ratio {uniq_ratio:.2f}")

        tasks = Counter(r["task"] for r in rows)
        for task, n in tasks.most_common():
            sub = [r for r in rows if r["task"] == task]
            keys = Counter(r.get("label_key") for r in sub)
            top_n = keys.most_common(1)[0][1] if keys else 0
            collapse = top_n / max(len(sub), 1)
            print(f"  [{task}] n={len(sub)} keys={dict(keys.most_common(8))} "
                  f"collapse={collapse:.3f}")
            if task in ALERT_TASKS and collapse > args.max_alert_collapse:
                hard_fails.append(
                    f"{gym}/{task}: alert collapse {collapse:.2f} > "
                    f"{args.max_alert_collapse} keys={dict(keys)}"
                )
            if len(keys) == 1:
                hard_fails.append(f"{gym}/{task}: single label_key {list(keys)[0]!r}")

    print("\n===== WARNINGS =====")
    for w in warns or ["(none)"]:
        print(w)
    print("\n===== HARD FAILS =====")
    for f in hard_fails or ["(none)"]:
        print(f)

    if hard_fails:
        print("\nAUDIT_FAIL")
        raise SystemExit(1)
    print("\nAUDIT_PASS")


if __name__ == "__main__":
    main()
