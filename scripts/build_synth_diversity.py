#!/usr/bin/env python3
"""Build synth-diversity extras (CA + word games + NLP cloze) for Phase 2→6 mixes.

Default caps keep these as diversity axes, not a full replace of phase2_train:
  - cellular_automata: 12_000 train / 400 eval
  - word_games: 12_000 train / 400 eval
  - nlp_cloze: 12_000 train / 400 eval  (WikiText-2 doc-disjoint)

Does not overwrite phase2_train_200k.jsonl unless --merge-into-phase2 is set.

When merging, existing rows for the same gyms are replaced (not double-appended).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from systemone_lite.synth.cellular_automata import generate_ca_samples
from systemone_lite.synth.nlp_cloze import generate_nlp_cloze_samples
from systemone_lite.synth.word_games import generate_word_game_samples

ROOT = Path(__file__).resolve().parents[1]

DIVERSITY_GYMS = frozenset({"cellular_automata", "word_games", "nlp_cloze"})


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ensure_gym(row: dict, default: str) -> dict:
    meta = dict(row.get("meta") or {})
    meta["gym"] = meta.get("gym") or default
    out = dict(row)
    out["meta"] = meta
    return out


def validate(rows: list[dict], name: str) -> None:
    forbidden = ["RECOMMENDED", "OPTIMAL", "CAPTURES enemy", "DELIVERS CHECK"]
    gyms: Counter[str] = Counter()
    for i, r in enumerate(rows):
        for key in ("task", "state", "instructions", "criteria", "label_alias", "label_key"):
            if key not in r:
                raise ValueError(f"{name} row {i} missing {key}")
        if r["label_alias"] not in r["criteria"]:
            raise ValueError(f"{name} row {i} bad label_alias")
        dump = json.dumps(r, ensure_ascii=False)
        for kw in forbidden:
            if kw in dump:
                raise ValueError(f"{name} row {i} forbidden {kw!r}")
        gyms[str((r.get("meta") or {}).get("gym"))] += 1
    print(f"✅ {name}: n={len(rows)} gyms={dict(gyms)}")


def strip_diversity(rows: list[dict]) -> list[dict]:
    return [
        r
        for r in rows
        if str((r.get("meta") or {}).get("gym") or "") not in DIVERSITY_GYMS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca-train", type=int, default=12_000)
    parser.add_argument("--word-train", type=int, default=12_000)
    parser.add_argument("--cloze-train", type=int, default=12_000)
    parser.add_argument("--ca-eval", type=int, default=400)
    parser.add_argument("--word-eval", type=int, default=400)
    parser.add_argument("--cloze-eval", type=int, default=400)
    parser.add_argument(
        "--train-out",
        type=Path,
        default=ROOT / "data" / "synth_diversity_train.jsonl",
    )
    parser.add_argument(
        "--eval-out",
        type=Path,
        default=ROOT / "data" / "synth_diversity_eval.jsonl",
    )
    parser.add_argument(
        "--merge-into-phase2",
        action="store_true",
        help="Replace diversity gyms in phase2_train/eval (shuffled rewrite)",
    )
    parser.add_argument(
        "--phase2-train",
        type=Path,
        default=ROOT / "data" / "phase2_train_200k.jsonl",
    )
    parser.add_argument(
        "--phase2-eval",
        type=Path,
        default=ROOT / "data" / "phase2_eval_4k.jsonl",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("=== Generating cellular automata ===", flush=True)
    ca_train = [
        ensure_gym(s.to_json(), "cellular_automata")
        for s in generate_ca_samples(args.ca_train, seed=args.seed)
    ]
    ca_eval = [
        ensure_gym(s.to_json(), "cellular_automata")
        for s in generate_ca_samples(args.ca_eval, seed=args.seed + 101)
    ]
    print(f"  ca train={len(ca_train)} eval={len(ca_eval)}", flush=True)

    print("=== Generating word games (TRAIN/EVAL vocab disjoint) ===", flush=True)
    word_train = [
        ensure_gym(s.to_json(), "word_games")
        for s in generate_word_game_samples(
            args.word_train, seed=args.seed + 3, split="train"
        )
    ]
    word_eval = [
        ensure_gym(s.to_json(), "word_games")
        for s in generate_word_game_samples(
            args.word_eval, seed=args.seed + 303, split="eval"
        )
    ]
    print(f"  word train={len(word_train)} eval={len(word_eval)}", flush=True)

    print("=== Generating NLP cloze (WikiText-2 doc-disjoint) ===", flush=True)
    cloze_train = [
        ensure_gym(s.to_json(), "nlp_cloze")
        for s in generate_nlp_cloze_samples(
            args.cloze_train, seed=args.seed + 5, split="train"
        )
    ]
    cloze_eval = [
        ensure_gym(s.to_json(), "nlp_cloze")
        for s in generate_nlp_cloze_samples(
            args.cloze_eval, seed=args.seed + 505, split="eval"
        )
    ]
    print(f"  cloze train={len(cloze_train)} eval={len(cloze_eval)}", flush=True)

    train = ca_train + word_train + cloze_train
    eval_rows = ca_eval + word_eval + cloze_eval
    rng.shuffle(train)
    rng.shuffle(eval_rows)

    validate(train, "synth_diversity_train")
    validate(eval_rows, "synth_diversity_eval")
    write_jsonl(args.train_out, train)
    write_jsonl(args.eval_out, eval_rows)
    print(f"Wrote {len(train)} → {args.train_out}")
    print(f"Wrote {len(eval_rows)} → {args.eval_out}")

    if args.merge_into_phase2:
        if not args.phase2_train.exists():
            raise SystemExit(f"missing {args.phase2_train}")
        base_train = strip_diversity(load_jsonl(args.phase2_train))
        base_eval = strip_diversity(load_jsonl(args.phase2_eval))
        merged_train = base_train + train
        merged_eval = base_eval + eval_rows
        rng.shuffle(merged_train)
        rng.shuffle(merged_eval)
        validate(merged_train, "phase2+diversity train")
        validate(merged_eval, "phase2+diversity eval")
        write_jsonl(args.phase2_train, merged_train)
        write_jsonl(args.phase2_eval, merged_eval)
        print(
            f"Merged → train {len(merged_train)} ({args.phase2_train}), "
            f"eval {len(merged_eval)} ({args.phase2_eval})"
        )


if __name__ == "__main__":
    main()
