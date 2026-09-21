#!/usr/bin/env python3
"""Honest Phase 2 choice eval: option-alias shuffle + per-gym / per-task metrics."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from systemone_lite.infer import DEFAULT_MODEL_ID
from systemone_lite.prompt import build_prompt
from systemone_lite.schema import ChoiceQuestion

ROOT = Path(__file__).resolve().parents[1]


def encode_alias_token_id(tokenizer, alias: str) -> int:
    for candidate in (alias, f" {alias}", f"\n{alias}"):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    ids = tokenizer.encode(alias, add_special_tokens=False)
    return ids[0]


def reshuffle_aliases(
    rng: random.Random,
    criteria: dict[str, str],
    label_alias: str,
) -> tuple[dict[str, str], str]:
    """Reassign A/B/C… aliases so letter/order bias cannot inflate accuracy."""
    if label_alias not in criteria:
        raise KeyError(f"label_alias {label_alias!r} missing from criteria")
    items = list(criteria.items())
    rng.shuffle(items)
    new_criteria: dict[str, str] = {}
    new_label: str | None = None
    for i, (old_alias, text) in enumerate(items):
        alias = chr(ord("A") + i) if i < 26 else str(i)
        new_criteria[alias] = text
        if old_alias == label_alias:
            new_label = alias
    if new_label is None:
        raise RuntimeError("label alias lost during reshuffle")
    return new_criteria, new_label


@torch.inference_mode()
def evaluate(
    *,
    data_path: Path,
    model_id: str,
    limit: int,
    max_length: int,
    shuffle: bool,
    seed: int,
    gym_filter: set[str] | None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    rows: list[dict] = []
    with data_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            gym = str((row.get("meta") or {}).get("gym") or "unknown")
            if gym_filter is not None and gym not in gym_filter:
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break

    by_task: dict[str, list[int]] = defaultdict(list)
    by_gym: dict[str, list[int]] = defaultdict(list)
    correct = 0

    for idx, row in enumerate(rows):
        criteria = dict(row["criteria"])
        label_alias = row["label_alias"]
        if shuffle:
            criteria, label_alias = reshuffle_aliases(
                random.Random(seed + idx), criteria, label_alias
            )
        question = ChoiceQuestion(
            type="choice",
            instructions=row["instructions"],
            criteria=criteria,
        )
        prompt = build_prompt(row["state"], question)
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**enc)
        last = enc["attention_mask"].sum(dim=1) - 1
        logits = out.logits[0, last[0]]
        aliases = list(criteria.keys())
        alias_ids = [encode_alias_token_id(tokenizer, a) for a in aliases]
        pred_idx = int(logits[alias_ids].argmax().item())
        pred = aliases[pred_idx]
        hit = int(pred == label_alias)
        correct += hit
        by_task[row["task"]].append(hit)
        gym = str((row.get("meta") or {}).get("gym") or "unknown")
        by_gym[gym].append(hit)

    def pack(groups: dict[str, list[int]]) -> dict[str, dict[str, float | int]]:
        return {
            key: {"n": len(hits), "accuracy": sum(hits) / max(len(hits), 1)}
            for key, hits in sorted(groups.items())
        }

    return {
        "model": model_id,
        "data": str(data_path),
        "n": len(rows),
        "accuracy": correct / max(len(rows), 1),
        "shuffle_aliases": shuffle,
        "seed": seed,
        "per_gym": pack(by_gym),
        "per_task": pack(by_task),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 honest choice evaluation")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep stored alias order (not recommended for claims)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gyms",
        default="",
        help="Comma-separated gym filter (empty = all)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    raw_gyms = {g.strip() for g in args.gyms.split(",") if g.strip()}
    gym_filter = raw_gyms or None

    report = evaluate(
        data_path=args.data,
        model_id=args.model,
        limit=args.limit,
        max_length=args.max_length,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        gym_filter=gym_filter,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
