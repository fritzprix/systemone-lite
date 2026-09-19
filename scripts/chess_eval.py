#!/usr/bin/env python3
"""Evaluate move-choice accuracy on a chess JSONL split."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--task", default="move")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=1024)
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

    rows = []
    with args.data.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["task"] == args.task:
                rows.append(row)
            if len(rows) >= args.limit:
                break

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
        correct += int(pred == row["label_alias"])

    acc = correct / max(len(rows), 1)
    print(json.dumps({"n": len(rows), "task": args.task, "accuracy": acc}, indent=2))


if __name__ == "__main__":
    main()
