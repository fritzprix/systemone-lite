#!/usr/bin/env python3
"""Evaluate choice accuracy on a general System One JSONL split."""

from __future__ import annotations

import argparse
import json
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


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    rows: list[dict] = []
    with args.data.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    by_task: dict[str, list[int]] = defaultdict(list)
    correct = 0
    for row in rows:
        question = ChoiceQuestion(
            type="choice",
            instructions=row["instructions"],
            criteria=row["criteria"],
        )
        prompt = build_prompt(row["state"], question)
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        ).to(device)
        out = model(**enc)
        last = enc["attention_mask"].sum(dim=1) - 1
        logits = out.logits[0, last[0]]
        aliases = list(row["criteria"].keys())
        alias_ids = [encode_alias_token_id(tokenizer, a) for a in aliases]
        pred_idx = int(logits[alias_ids].argmax().item())
        pred = aliases[pred_idx]
        hit = int(pred == row["label_alias"])
        correct += hit
        by_task[row["task"]].append(hit)

    per_task = {
        task: {"n": len(hits), "accuracy": sum(hits) / max(len(hits), 1)}
        for task, hits in sorted(by_task.items())
    }
    report = {
        "model": args.model,
        "n": len(rows),
        "accuracy": correct / max(len(rows), 1),
        "per_task": per_task,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
