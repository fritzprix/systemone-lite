#!/usr/bin/env python3
"""Supervised fine-tune Qwen for chess System One choice answers."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from systemone_lite.infer import DEFAULT_MODEL_ID, reset_engine
from systemone_lite.prompt import build_prompt
from systemone_lite.schema import ChoiceQuestion

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Example:
    prompt: str
    label_alias: str


class ChessJsonlDataset(Dataset[Example]):
    def __init__(self, path: Path, tasks: set[str] | None = None) -> None:
        self.rows: list[Example] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if tasks is not None and row["task"] not in tasks:
                    continue
                question = ChoiceQuestion(
                    type="choice",
                    instructions=row["instructions"],
                    criteria=row["criteria"],
                )
                prompt = build_prompt(row["state"], question)
                self.rows.append(Example(prompt=prompt, label_alias=row["label_alias"]))
        if not self.rows:
            raise ValueError(f"no examples loaded from {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Example:
        return self.rows[idx]


def encode_alias_token_id(tokenizer, alias: str) -> int:
    for candidate in (alias, f" {alias}", f"\n{alias}"):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    ids = tokenizer.encode(alias, add_special_tokens=False)
    if not ids:
        raise ValueError(f"cannot tokenize alias {alias!r}")
    return ids[0]


def collate(batch: list[Example], tokenizer, max_length: int, device: torch.device):
    prompts = [ex.prompt for ex in batch]
    labels = [encode_alias_token_id(tokenizer, ex.label_alias) for ex in batch]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    label_ids = torch.tensor(labels, dtype=torch.long, device=device)
    return input_ids, attention_mask, label_ids


def train(
    *,
    data_path: Path,
    output_dir: Path,
    model_id: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    max_steps: int | None,
    seed: int,
    tasks: set[str],
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prefer bf16 when available; else fp16 + GradScaler (fp32 OOMs on 12GB).
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device.type == "cuda" and not use_bf16
    load_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=load_dtype, trust_remote_code=True
    )
    model.to(device)
    model.train()

    dataset = ChessJsonlDataset(data_path, tasks=tasks)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer, max_length, device),
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
    )
    total_steps = epochs * len(loader)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    step = 0
    running = 0.0
    for epoch in range(epochs):
        for input_ids, attention_mask, label_ids in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if use_bf16 else torch.float16,
                enabled=(use_bf16 or use_fp16),
            ):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                last_idx = attention_mask.sum(dim=1) - 1
                logits = outputs.logits[
                    torch.arange(input_ids.size(0), device=device),
                    last_idx,
                ].float()
                loss = torch.nn.functional.cross_entropy(logits, label_ids)

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step+1}: {loss.item()}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            running += float(loss.item())
            if step % 10 == 0 or step == 1:
                print(
                    f"epoch={epoch+1} step={step}/{total_steps} "
                    f"loss={loss.item():.4f} avg={running/step:.4f}"
                )
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    meta = {
        "base_model": model_id,
        "data": str(data_path),
        "steps": step,
        "epochs": epochs,
        "tasks": sorted(tasks),
        "final_avg_loss": running / max(step, 1),
    }
    (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved adapter/model → {output_dir}")
    reset_engine()  # force reload if server uses singleton


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune chess System One policy")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "chess_distill.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "checkpoints" / "chess-sft",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tasks",
        default="move,piece,destination",
        help="Comma-separated tasks to include",
    )
    args = parser.parse_args()

    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    train(
        data_path=args.data,
        output_dir=args.out,
        model_id=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        max_steps=args.max_steps,
        seed=args.seed,
        tasks=tasks,
    )


if __name__ == "__main__":
    main()
