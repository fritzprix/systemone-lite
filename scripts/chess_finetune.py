#!/usr/bin/env python3
"""Supervised fine-tune Qwen for System One choice answers (chess or general)."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from systemone_lite.infer import DEFAULT_MODEL_ID, reset_engine
from systemone_lite.prompt import build_prompt
from systemone_lite.schema import ChoiceQuestion

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Example:
    prompt: str
    label_alias: str
    gym: str
    task: str


class ChoiceJsonlDataset(Dataset[Example]):
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
                meta = row.get("meta") or {}
                gym = str(meta.get("gym") or row["task"].split(".", 1)[0])
                self.rows.append(
                    Example(
                        prompt=prompt,
                        label_alias=row["label_alias"],
                        gym=gym,
                        task=row["task"],
                    )
                )
        if not self.rows:
            raise ValueError(f"no examples loaded from {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Example:
        return self.rows[idx]


class StratifiedGymBatchSampler(Sampler[list[int]]):
    """Build batches that mix gyms (round-robin within each batch when possible)."""

    def __init__(
        self,
        gyms: list[str],
        *,
        batch_size: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self._by_gym: dict[str, list[int]] = defaultdict(list)
        for idx, gym in enumerate(gyms):
            self._by_gym[gym].append(idx)
        self._gym_names = sorted(self._by_gym)
        if not self._gym_names:
            raise ValueError("no gym labels for stratified sampler")

    def __iter__(self):
        rng = random.Random(self.seed)
        pools = {g: list(idxs) for g, idxs in self._by_gym.items()}
        for idxs in pools.values():
            rng.shuffle(idxs)

        batch: list[int] = []
        gym_cycle = list(self._gym_names)
        while True:
            progress = False
            rng.shuffle(gym_cycle)
            for gym in gym_cycle:
                if not pools[gym]:
                    continue
                batch.append(pools[gym].pop())
                progress = True
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if not progress:
                break
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        n = sum(len(v) for v in self._by_gym.values())
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


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
    tasks: set[str] | None,
    stratified: bool,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device.type == "cuda" and not use_bf16
    load_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=load_dtype, trust_remote_code=True
    )
    model.to(device)
    model.train()

    dataset = ChoiceJsonlDataset(data_path, tasks=tasks)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
    )

    steps_per_epoch = (len(dataset) + batch_size - 1) // batch_size
    total_steps = epochs * steps_per_epoch
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=max(1, total_steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    gym_hist: dict[str, int] = defaultdict(int)
    step = 0
    running = 0.0
    print(
        f"train n={len(dataset)} stratified={stratified} "
        f"batch={batch_size} epochs={epochs} steps≈{total_steps}"
    )

    for epoch in range(epochs):
        if stratified:
            index_batches = list(
                StratifiedGymBatchSampler(
                    [ex.gym for ex in dataset.rows],
                    batch_size=batch_size,
                    seed=seed + epoch,
                )
            )
        else:
            indices = list(range(len(dataset)))
            random.Random(seed + epoch).shuffle(indices)
            index_batches = [
                indices[i : i + batch_size]
                for i in range(0, len(indices), batch_size)
            ]

        for index_batch in index_batches:
            examples = [dataset[i] for i in index_batch]
            for ex in examples:
                gym_hist[ex.gym] += 1
            input_ids, attention_mask, label_ids = collate(
                examples, tokenizer, max_length, device
            )
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
                raise RuntimeError(f"non-finite loss at step {step + 1}: {loss.item()}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            running += float(loss.item())
            if step % 50 == 0 or step == 1:
                print(
                    f"epoch={epoch + 1} step={step}/{total_steps} "
                    f"loss={loss.item():.4f} avg={running / step:.4f}"
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
        "tasks": sorted(tasks) if tasks is not None else ["all"],
        "stratified": stratified,
        "batch_size": batch_size,
        "final_avg_loss": running / max(step, 1),
        "samples_by_gym": dict(gym_hist),
    }
    (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved adapter/model → {output_dir}")
    print("samples_by_gym:", dict(gym_hist))
    reset_engine()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune System One choice policy")
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
        help="Comma-separated tasks, or 'all' for every task in the JSONL",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Mix gyms inside each batch (recommended for general distill)",
    )
    args = parser.parse_args()

    raw_tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    tasks: set[str] | None
    if not raw_tasks or raw_tasks == {"all"}:
        tasks = None
    else:
        tasks = raw_tasks

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
        stratified=args.stratified,
    )


if __name__ == "__main__":
    main()
