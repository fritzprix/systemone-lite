#!/usr/bin/env python3
"""Supervised fine-tune Qwen for System One choice answers (chess or general).

Supports mid-run checkpoints + resume so long Phase 2 jobs survive interrupts:
  python scripts/chess_finetune.py ... --out checkpoints/systemone-spatial-v2 \\
      --save-every 1000 --resume
  # optional W&B: --wandb --wandb-project systemone-lite

Schedule horizon (--max-steps / epochs) is immutable within a resume chain.
Plan the full run length up front; extending mid-run is a new experiment (see #8).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from systemone_lite.infer import DEFAULT_MODEL_ID, reset_engine
from systemone_lite.prompt import build_prompt
from systemone_lite.schema import ChoiceQuestion

ROOT = Path(__file__).resolve().parents[1]
LAST_DIRNAME = "last"
STATE_NAME = "train_state.pt"


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
    """Build batches that mix gyms (round-robin with replacement when small gyms deplete)."""

    def __init__(
        self,
        gyms: list[str],
        *,
        batch_size: int,
        seed: int,
        drop_last: bool = False,
        total_samples: int | None = None,
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
        self.total_samples = total_samples if total_samples is not None else len(gyms)

    def __iter__(self):
        rng = random.Random(self.seed)
        pools = {}
        for g, idxs in self._by_gym.items():
            pools[g] = list(idxs)
            rng.shuffle(pools[g])

        batch: list[int] = []
        gym_cycle = list(self._gym_names)
        yielded_samples = 0

        while yielded_samples < self.total_samples:
            rng.shuffle(gym_cycle)
            for gym in gym_cycle:
                if not pools[gym]:
                    pools[gym] = list(self._by_gym[gym])
                    rng.shuffle(pools[gym])
                batch.append(pools[gym].pop())
                yielded_samples += 1
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                if yielded_samples >= self.total_samples:
                    break
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.total_samples // self.batch_size
        return (self.total_samples + self.batch_size - 1) // self.batch_size


def encode_alias_token_id(tokenizer, alias: str) -> int:
    for candidate in (alias, f" {alias}", f"\n{alias}"):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    ids = tokenizer.encode(alias, add_special_tokens=False)
    if not ids:
        raise ValueError(f"cannot tokenize alias {alias!r}")
    return ids[0]


def run_validation(
    *,
    model,
    tokenizer,
    eval_path: Path,
    device: torch.device,
    max_length: int,
    limit: int,
    seed: int,
) -> dict[str, float | int | dict[str, float]]:
    """Held-out choice accuracy with alias reshuffle (honest mid-train probe)."""
    rows: list[dict] = []
    with eval_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    if not rows:
        return {"n": 0, "accuracy": 0.0, "per_gym": {}}

    model.eval()
    by_gym: dict[str, list[int]] = defaultdict(list)
    correct = 0
    with torch.inference_mode():
        for idx, row in enumerate(rows):
            criteria = dict(row["criteria"])
            label_alias = row["label_alias"]
            items = list(criteria.items())
            random.Random(seed + idx).shuffle(items)
            new_criteria: dict[str, str] = {}
            new_label: str | None = None
            for i, (old_alias, text) in enumerate(items):
                alias = chr(ord("A") + i) if i < 26 else str(i)
                new_criteria[alias] = text
                if old_alias == label_alias:
                    new_label = alias
            if new_label is None:
                continue
            question = ChoiceQuestion(
                type="choice",
                instructions=row["instructions"],
                criteria=new_criteria,
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
            aliases = list(new_criteria.keys())
            alias_ids = [encode_alias_token_id(tokenizer, a) for a in aliases]
            pred = aliases[int(logits[alias_ids].argmax().item())]
            hit = int(pred == new_label)
            correct += hit
            gym = str((row.get("meta") or {}).get("gym") or "unknown")
            by_gym[gym].append(hit)
    model.train()
    per_gym = {
        g: sum(hits) / max(len(hits), 1) for g, hits in sorted(by_gym.items())
    }
    return {
        "n": len(rows),
        "accuracy": correct / max(len(rows), 1),
        "per_gym": per_gym,
    }


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


def _config_fingerprint(
    *,
    data_path: Path,
    model_id: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    max_steps: int | None,
    seed: int,
    tasks: set[str] | None,
    stratified: bool,
    grad_accum: int,
    warmup_steps: int | None,
    scheduler_name: str,
) -> dict[str, Any]:
    return {
        "data": str(data_path.resolve()),
        "model_id": model_id,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "max_length": max_length,
        "max_steps": max_steps,
        "seed": seed,
        "tasks": sorted(tasks) if tasks is not None else ["all"],
        "stratified": stratified,
        "grad_accum": grad_accum,
        "warmup_steps": warmup_steps,
        "scheduler": scheduler_name,
    }


def build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    warmup_steps: int,
    total_steps: int,
):
    name = scheduler_name.lower().strip()
    warm = max(0, int(warmup_steps))
    total = max(1, int(total_steps))
    if name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, warm) if warm > 0 else max(1, total // 10),
            num_training_steps=total,
        )
    if name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm,
            num_training_steps=total,
        )
    if name in {"constant", "constant_with_warmup"}:
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm,
        )
    raise SystemExit(f"unknown --scheduler {scheduler_name!r} (linear|cosine|constant)")


def _last_dir(output_dir: Path) -> Path:
    return output_dir / LAST_DIRNAME


def _state_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / STATE_NAME


def save_checkpoint(
    *,
    ckpt_dir: Path,
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    step: int,
    epoch: int,
    batch_in_epoch: int,
    running: float,
    gym_hist: dict[str, int],
    total_steps: int,
    config: dict[str, Any],
    wandb_id: str | None = None,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "running": running,
            "gym_hist": dict(gym_hist),
            "total_steps": total_steps,
            "config": config,
            "wandb_id": wandb_id,
            "last_lr": current_optimizer_lr(optimizer),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "py_rng": random.getstate(),
        },
        _state_path(ckpt_dir),
    )
    meta = {
        "step": step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "total_steps": total_steps,
        "avg_loss": running / max(step, 1),
        "last_lr": current_optimizer_lr(optimizer),
        "config": config,
    }
    (ckpt_dir / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"checkpoint → {ckpt_dir} (step={step}/{total_steps})", flush=True)


def load_checkpoint_state(ckpt_dir: Path) -> dict[str, Any]:
    path = _state_path(ckpt_dir)
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def prune_step_checkpoints(output_dir: Path, *, keep: int) -> None:
    if keep < 0:
        return
    step_dirs = sorted(
        (p for p in output_dir.glob("step-*") if p.is_dir() and not p.is_symlink()),
        key=lambda p: int(p.name.split("-", 1)[1]),
    )
    for old in step_dirs[:-keep] if keep else step_dirs:
        shutil.rmtree(old, ignore_errors=True)


def point_last_at(output_dir: Path, step_dir: Path) -> Path:
    """Make out/last a symlink to step-N (no second full copy)."""
    last = _last_dir(output_dir)
    if last.is_symlink() or last.exists():
        if last.is_symlink() or last.is_file():
            last.unlink()
        else:
            shutil.rmtree(last)
    last.symlink_to(step_dir.name, target_is_directory=True)
    return last


def save_rotating_checkpoint(
    *,
    output_dir: Path,
    step: int,
    keep_checkpoints: int,
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    batch_in_epoch: int,
    running: float,
    gym_hist: dict[str, int],
    total_steps: int,
    config: dict[str, Any],
    wandb_id: str | None = None,
) -> Path:
    """Write step-N, rotate to keep N snapshots, point last/ → newest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    step_dir = output_dir / f"step-{step}"
    if step_dir.exists() or step_dir.is_symlink():
        if step_dir.is_symlink() or step_dir.is_file():
            step_dir.unlink()
        else:
            shutil.rmtree(step_dir)
    save_checkpoint(
        ckpt_dir=step_dir,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=step,
        epoch=epoch,
        batch_in_epoch=batch_in_epoch,
        running=running,
        gym_hist=gym_hist,
        total_steps=total_steps,
        config=config,
        wandb_id=wandb_id,
    )
    prune_step_checkpoints(output_dir, keep=keep_checkpoints)
    # newest may have been pruned only if keep==0; otherwise point last at it
    if step_dir.exists():
        point_last_at(output_dir, step_dir)
    return step_dir



def load_best_val_acc(output_dir: Path) -> float | None:
    meta_path = output_dir / "best_val" / "best_val_meta.json"
    if not meta_path.exists():
        return None
    try:
        raw = json.loads(meta_path.read_text())
        return float(raw["val_accuracy"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def maybe_save_best_val(
    *,
    output_dir: Path,
    model,
    tokenizer,
    step: int,
    accuracy: float,
    best_acc: float | None,
) -> float:
    """Keep a non-rotated copy of the best mid-train val checkpoint (#9)."""
    if best_acc is not None and accuracy <= best_acc:
        return best_acc
    best_dir = output_dir / "best_val"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    (best_dir / "best_val_meta.json").write_text(
        json.dumps({"step": step, "val_accuracy": accuracy}, indent=2) + "\n"
    )
    print(
        f"best_val → {best_dir} (step={step} acc={accuracy:.4f})",
        flush=True,
    )
    return accuracy


def warn_if_schedule_exhausted(*, lr: float, step: int, total_steps: int) -> None:
    """Loud alarm when a run ends at ~zero LR (yesterday's failure mode)."""
    if step < total_steps:
        return
    if lr > 1e-10:
        return
    print(
        "WARNING: schedule exhausted — final train/lr≈0 at end of horizon. "
        "This is a finished schedule, NOT a mid-run checkpoint. "
        "Do not raise --max-steps and --resume the same run (GitHub #8). "
        f"step={step}/{total_steps} lr={lr:.3e}",
        flush=True,
    )


def maybe_init_wandb(
    *,
    enabled: bool,
    project: str,
    run_name: str | None,
    config: dict[str, Any],
    resume_id: str | None,
) -> Any | None:
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb requested but not installed; pip install 'wandb' "
            "or omit --wandb"
        ) from exc
    init_kwargs: dict[str, Any] = {
        "project": project,
        "config": config,
        "resume": "allow",
    }
    if run_name:
        init_kwargs["name"] = run_name
    if resume_id:
        init_kwargs["id"] = resume_id
    return wandb.init(**init_kwargs)


def configs_compatible(saved: dict[str, Any], now: dict[str, Any]) -> bool:
    """Train hyperparams must match exactly — including max_steps.

    Changing the schedule horizon mid-run recomputes LambdaLR at the same
    global step and can jump LR by orders of magnitude (see GitHub #8).
    A longer/shorter horizon is a **new** experiment, not a resume.
    """
    return dict(saved) == dict(now)


def current_optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def assert_lr_continuity(
    *,
    saved_lr: float | None,
    current_lr: float,
    tol: float = 0.05,
) -> None:
    """Refuse resume when LR jumps more than ``tol`` (relative)."""
    if saved_lr is None:
        return
    denom = max(abs(float(saved_lr)), 1e-12)
    jump = abs(float(current_lr) - float(saved_lr)) / denom
    if jump > tol:
        raise SystemExit(
            f"Unsafe LR discontinuity on resume: {saved_lr} -> {current_lr} "
            f"(jump={jump:.1%}, tol={tol:.0%}). "
            "Refuse to continue — start a new run / W&B experiment if the "
            "schedule horizon must change (GitHub #8)."
        )


def assert_schedule_horizon(
    *,
    saved_total_steps: int | None,
    total_steps: int,
) -> None:
    if saved_total_steps is None:
        return
    if int(saved_total_steps) != int(total_steps):
        raise SystemExit(
            f"Unsafe schedule horizon change on resume: "
            f"checkpoint total_steps={saved_total_steps} vs now={total_steps}. "
            "Plan the full horizon before the first step; do not raise "
            "--max-steps mid-run (GitHub #8)."
        )


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
    save_every: int,
    keep_checkpoints: int,
    resume: bool,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_run_name: str | None,
    eval_data: Path | None,
    eval_every: int,
    eval_limit: int,
    grad_accum: int = 1,
    warmup_steps: int | None = None,
    scheduler_name: str = "linear",
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if grad_accum < 1:
        raise SystemExit("--grad-accum must be >= 1")

    config = _config_fingerprint(
        data_path=data_path,
        model_id=model_id,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        max_steps=max_steps,
        seed=seed,
        tasks=tasks,
        stratified=stratified,
        grad_accum=grad_accum,
        warmup_steps=warmup_steps,
        scheduler_name=scheduler_name,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device.type == "cuda" and not use_bf16
    load_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    last = _last_dir(output_dir)
    resume_dir: Path | None = None
    if resume and _state_path(last).exists():
        resume_dir = last
    elif resume:
        print(f"--resume set but no state at {last}; starting fresh", flush=True)

    if resume_dir is not None:
        state = load_checkpoint_state(resume_dir)
        saved_cfg = state.get("config") or {}
        if not configs_compatible(saved_cfg, config):
            raise SystemExit(
                "resume config mismatch vs current CLI args; "
                "refuse to continue.\n"
                "Note: --max-steps / schedule horizon must match the checkpoint "
                "(GitHub #8). Extending the horizon is a new experiment.\n"
                f"saved={saved_cfg}\nnow={config}"
            )
        model_source = str(resume_dir)
        print(
            f"resuming from {resume_dir} at step={state['step']} "
            f"epoch={state['epoch']} batch={state['batch_in_epoch']}",
            flush=True,
        )
    else:
        state = None
        model_source = model_id

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_source, dtype=load_dtype, trust_remote_code=True
    )
    model.to(device)
    model.train()
    # 12GB cards: activation memory spikes on long prompts otherwise OOM mid-run.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    dataset = ChoiceJsonlDataset(data_path, tasks=tasks)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
    )

    # Optimizer-step horizon (grad_accum micro-batches share one scheduler step).
    micro_per_epoch = (len(dataset) + batch_size - 1) // batch_size
    steps_per_epoch = (micro_per_epoch + grad_accum - 1) // grad_accum
    total_steps = epochs * steps_per_epoch
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    warm = (
        warmup_steps
        if warmup_steps is not None
        else max(1, total_steps // 10)
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        warmup_steps=warm,
        total_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    gym_hist: dict[str, int] = defaultdict(int)
    step = 0
    running = 0.0
    start_epoch = 0
    start_batch = 0
    wandb_id: str | None = None

    if state is not None:
        assert_schedule_horizon(
            saved_total_steps=state.get("total_steps"),
            total_steps=total_steps,
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        # LambdaLR lambdas are closures over the *new* horizon; force the
        # restored last_epoch through get_last_lr before the continuity check.
        _ = scheduler.get_last_lr()
        for group, restored_lr in zip(
            optimizer.param_groups, scheduler.get_last_lr(), strict=True
        ):
            group["lr"] = restored_lr
        assert_lr_continuity(
            saved_lr=state.get("last_lr"),
            current_lr=current_optimizer_lr(optimizer),
        )
        step = int(state["step"])
        running = float(state["running"])
        start_epoch = int(state["epoch"])
        start_batch = int(state["batch_in_epoch"])
        for gym, count in (state.get("gym_hist") or {}).items():
            gym_hist[gym] = int(count)
        if state.get("torch_rng") is not None:
            torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if state.get("py_rng") is not None:
            random.setstate(state["py_rng"])
        wandb_id = state.get("wandb_id")
        # Early --max-steps stops used to finalize with epoch=epochs, batch=0,
        # which makes range(start_epoch, epochs) empty. Repair from step count
        # only when the schedule horizon is unchanged (same --max-steps).
        if start_epoch >= epochs and step < total_steps:
            start_epoch = step // max(steps_per_epoch, 1)
            start_batch = step % max(steps_per_epoch, 1)
            print(
                f"resume cursor repaired → epoch={start_epoch} "
                f"batch={start_batch} (from step={step})",
                flush=True,
            )
        if step >= total_steps:
            print(f"already complete at step={step}; exporting final weights only")
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            return
        print(
            f"resume lr={current_optimizer_lr(optimizer):.6e} "
            f"(saved last_lr={state.get('last_lr')})",
            flush=True,
        )

    wb = maybe_init_wandb(
        enabled=wandb_enabled,
        project=wandb_project,
        run_name=wandb_run_name,
        config=config,
        resume_id=wandb_id,
    )
    if wb is not None:
        wandb_id = wb.id

    print(
        f"train n={len(dataset)} stratified={stratified} "
        f"batch={batch_size}x{grad_accum} sched={scheduler_name} "
        f"warmup={warm} lr={lr} epochs={epochs} opt_steps≈{total_steps} "
        f"save_every={save_every} eval_every={eval_every} "
        f"resume={resume_dir is not None}",
        flush=True,
    )

    done = False
    final_epoch = start_epoch
    final_batch = start_batch
    # Resume must restore running-best or a worse mid-val overwrites best_val (#9).
    best_val_acc: float | None = load_best_val_acc(output_dir)
    if best_val_acc is not None:
        print(f"restored best_val_acc={best_val_acc:.4f} from disk", flush=True)
    loss_ema: float | None = None
    ema_beta = 2.0 / (100.0 + 1.0)  # ~100-step EMA
    micro_in_accum = 0
    accum_loss = 0.0

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
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

        batch_start = start_batch if epoch == start_epoch else 0
        for batch_i, index_batch in enumerate(index_batches):
            if batch_i < batch_start:
                continue
            examples = [dataset[i] for i in index_batch]
            for ex in examples:
                gym_hist[ex.gym] += 1
            input_ids, attention_mask, label_ids = collate(
                examples, tokenizer, max_length, device
            )
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
                loss = loss / grad_accum

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at opt_step {step + 1}: {loss.item()}"
                )

            scaler.scale(loss).backward()
            micro_in_accum += 1
            accum_loss += float(loss.item()) * grad_accum

            next_batch = batch_i + 1
            next_epoch = epoch
            if next_batch >= len(index_batches):
                next_batch = 0
                next_epoch = epoch + 1
            final_epoch = next_epoch
            final_batch = next_batch

            if micro_in_accum < grad_accum:
                continue

            scaler.unscale_(optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            raw_loss = accum_loss / grad_accum
            running += raw_loss
            loss_ema = (
                raw_loss
                if loss_ema is None
                else (1.0 - ema_beta) * loss_ema + ema_beta * raw_loss
            )
            micro_in_accum = 0
            accum_loss = 0.0
            cur_lr = float(scheduler.get_last_lr()[0])

            if step % 50 == 0 or step == 1:
                print(
                    f"epoch={epoch + 1} step={step}/{total_steps} "
                    f"loss={raw_loss:.4f} ema={loss_ema:.4f} "
                    f"avg={running / step:.4f} lr={cur_lr:.3e} "
                    f"gnorm={grad_norm:.3f}",
                    flush=True,
                )
            if wb is not None:
                wb.log(
                    {
                        "train/loss_raw": raw_loss,
                        "train/loss": raw_loss,
                        "train/loss_ema_100": float(loss_ema),
                        "train/avg_loss": running / step,
                        "train/epoch": epoch + 1,
                        "train/lr": cur_lr,
                        "train/gradient_norm": grad_norm,
                        "train/effective_batch_size": batch_size * grad_accum,
                        "train/optimizer_step": step,
                    },
                    step=step,
                )

            if save_every > 0 and step % save_every == 0:
                save_rotating_checkpoint(
                    output_dir=output_dir,
                    step=step,
                    keep_checkpoints=keep_checkpoints,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=next_epoch,
                    batch_in_epoch=next_batch,
                    running=running,
                    gym_hist=dict(gym_hist),
                    total_steps=total_steps,
                    config=config,
                    wandb_id=wandb_id,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if (
                eval_data is not None
                and eval_every > 0
                and step % eval_every == 0
            ):
                val = run_validation(
                    model=model,
                    tokenizer=tokenizer,
                    eval_path=eval_data,
                    device=device,
                    max_length=max_length,
                    limit=eval_limit,
                    seed=seed,
                )
                print(
                    f"val step={step} n={val['n']} acc={val['accuracy']:.4f} "
                    f"per_gym={val['per_gym']}",
                    flush=True,
                )
                best_val_acc = maybe_save_best_val(
                    output_dir=output_dir,
                    model=model,
                    tokenizer=tokenizer,
                    step=step,
                    accuracy=float(val["accuracy"]),
                    best_acc=best_val_acc,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if wb is not None:
                    payload: dict[str, float] = {
                        "val/accuracy": float(val["accuracy"]),
                        "val/n": float(val["n"]),
                    }
                    if best_val_acc is not None:
                        payload["val/best_accuracy"] = float(best_val_acc)
                    per_gym = val["per_gym"]
                    if isinstance(per_gym, dict):
                        for gym, acc in per_gym.items():
                            payload[f"val/gym/{gym}"] = float(acc)
                    wb.log(payload, step=step)

            if max_steps is not None and step >= max_steps:
                done = True
                break
            if step >= total_steps:
                done = True
                break
        if done:
            break
        start_batch = 0

    final_lr = current_optimizer_lr(optimizer)
    warn_if_schedule_exhausted(lr=final_lr, step=step, total_steps=total_steps)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_rotating_checkpoint(
        output_dir=output_dir,
        step=step,
        keep_checkpoints=keep_checkpoints,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=final_epoch,
        batch_in_epoch=final_batch,
        running=running,
        gym_hist=dict(gym_hist),
        total_steps=total_steps,
        config=config,
        wandb_id=wandb_id,
    )

    meta = {
        "base_model": model_id,
        "data": str(data_path),
        "steps": step,
        "epochs": epochs,
        "tasks": sorted(tasks) if tasks is not None else ["all"],
        "stratified": stratified,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "scheduler": scheduler_name,
        "warmup_steps": warm,
        "lr": lr,
        "final_lr": final_lr,
        "final_avg_loss": running / max(step, 1),
        "best_val_accuracy": best_val_acc,
        "samples_by_gym": dict(gym_hist),
        "save_every": save_every,
        "wandb_id": wandb_id,
    }
    (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved adapter/model → {output_dir}")
    print("samples_by_gym:", dict(gym_hist))
    if wb is not None:
        wb.summary.update(
            {
                "final_avg_loss": meta["final_avg_loss"],
                "steps": step,
                "final_lr": final_lr,
                "best_val_accuracy": best_val_acc,
            }
        )
        wb.finish()
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
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch-size × this)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Scheduler warmup optimizer steps (default: 10%% of horizon for linear)",
    )
    parser.add_argument(
        "--scheduler",
        default="linear",
        choices=("linear", "cosine", "constant"),
        help="LR schedule over the planned optimizer-step horizon",
    )
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
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Write resumable checkpoint under out/last every N steps (0=disable)",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        help="Rotate this many out/step-N dirs (out/last → symlink to newest). -1=all, 0=none",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from out/last/train_state.pt if present",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log metrics to Weights & Biases",
    )
    parser.add_argument("--wandb-project", default="systemone-lite")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=ROOT / "data" / "phase2_eval_4k.jsonl",
        help="Held-out JSONL for rare mid-train validation (empty path disables)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5000,
        help="Run validation every N steps (0=disable). Keep high — validation is slow.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=800,
        help="Max eval rows per validation pass (speed vs noise)",
    )
    args = parser.parse_args()

    raw_tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    tasks: set[str] | None
    if not raw_tasks or raw_tasks == {"all"}:
        tasks = None
    else:
        tasks = raw_tasks

    eval_data = args.eval_data if args.eval_data and str(args.eval_data) != "-" else None

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
        save_every=args.save_every,
        keep_checkpoints=args.keep_checkpoints,
        resume=args.resume,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        eval_data=eval_data,
        eval_every=args.eval_every,
        eval_limit=args.eval_limit,
        grad_accum=args.grad_accum,
        warmup_steps=args.warmup_steps,
        scheduler_name=args.scheduler,
    )


if __name__ == "__main__":
    main()
