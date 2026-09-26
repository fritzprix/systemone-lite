#!/usr/bin/env python3
"""Comprehensive training health & W&B monitor for systemone-lite.

Inspects local processes, GPU thermals/VRAM, local log files,
and queries W&B API / local artifacts to produce a diagnostic health report.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def get_gpu_status() -> dict:
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        parts = [p.strip() for p in res.stdout.strip().split(",")]
        return {
            "gpu_util_pct": float(parts[0]),
            "memory_used_mb": float(parts[1]),
            "memory_total_mb": float(parts[2]),
            "temperature_c": float(parts[3]),
            "power_w": float(parts[4]),
        }
    except Exception as e:
        return {"error": str(e)}


def get_running_process() -> dict:
    try:
        res = subprocess.run(
            ["pgrep", "-f", "chess_finetune.py"],
            capture_output=True,
            text=True,
        )
        pids = res.stdout.strip().split()
        if pids:
            # get process detail
            pinfo = subprocess.run(
                ["ps", "-o", "pid,user,%cpu,%mem,etime,cmd", "-p", pids[0]],
                capture_output=True,
                text=True,
            ).stdout.strip()
            return {"running": True, "pids": pids, "info": pinfo}
        return {"running": False, "pids": []}
    except Exception as e:
        return {"running": False, "error": str(e)}


def get_latest_local_log_metrics() -> dict:
    log_dir = REPO_ROOT / "logs"
    if not log_dir.exists():
        return {}
    log_files = sorted(
        log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True
    )
    relevant = [f for f in log_files if "pilot" in f.name or "train" in f.name]
    if not relevant:
        return {}
    latest_file = relevant[0]

    recent_steps = []
    recent_vals = []
    with open(latest_file, "r", errors="ignore") as f:
        lines = f.readlines()[-300:]

    for line in lines:
        if "step=" in line and "loss=" in line:
            m = re.search(
                r"step=(\d+)/(\d+)\s+loss=([0-9.]+)\s+ema=([0-9.]+)\s+avg=([0-9.]+)\s+lr=([0-9.e+-]+)\s+gnorm=([0-9.]+)",
                line,
            )
            if m:
                recent_steps.append(
                    {
                        "step": int(m.group(1)),
                        "max_steps": int(m.group(2)),
                        "loss": float(m.group(3)),
                        "ema": float(m.group(4)),
                        "avg": float(m.group(5)),
                        "lr": float(m.group(6)),
                        "gnorm": float(m.group(7)),
                    }
                )
        if "val/acc" in line or "val accuracy" in line or "best_val" in line:
            recent_vals.append(line.strip())

    return {
        "file": str(latest_file.relative_to(REPO_ROOT)),
        "recent_steps": recent_steps[-5:],
        "recent_vals": recent_vals[-5:],
    }


def get_wandb_status() -> dict:
    try:
        import wandb

        api = wandb.Api()
        # Find latest run in project doodream/systemone-lite
        runs = api.runs("doodream/systemone-lite", order="-created_at", per_page=1)
        if not runs:
            return {"status": "no_runs_found"}

        run = runs[0]
        summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
        gym_accuracies = {
            k.replace("val/gym/", ""): v
            for k, v in summary.items()
            if k.startswith("val/gym/")
        }

        return {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "url": run.url,
            "total_steps": run.lastHistoryStep,
            "summary": {
                "step": summary.get("train/optimizer_step"),
                "raw_loss": summary.get("train/loss_raw") or summary.get("train/loss"),
                "ema_loss": summary.get("train/loss_ema_100"),
                "avg_loss": summary.get("train/avg_loss"),
                "lr": summary.get("train/lr"),
                "gnorm": summary.get("train/gradient_norm"),
                "val_acc": summary.get("val/accuracy"),
                "best_val_acc": summary.get("val/best_accuracy"),
            },
            "gym_accuracies": gym_accuracies,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def evaluate_health(proc: dict, gpu: dict, wdb: dict, local: dict) -> list[str]:
    alerts = []

    # Process check
    if not proc.get("running"):
        alerts.append("ℹ️ [PROCESS] No active training process currently executing.")
    else:
        alerts.append("✅ [PROCESS] Training process is actively running.")

    # GPU check
    if "error" not in gpu:
        vram_free = gpu["memory_total_mb"] - gpu["memory_used_mb"]
        if vram_free < 1000 and proc.get("running"):
            alerts.append(f"⚠️ [GPU WARNING] VRAM free is low ({vram_free:.0f} MiB). Risk of OOM.")
        elif gpu["temperature_c"] > 83:
            alerts.append(f"⚠️ [GPU WARNING] High temperature ({gpu['temperature_c']}°C). Check cooling.")
        else:
            alerts.append(f"✅ [GPU OK] Util {gpu['gpu_util_pct']}%, Free VRAM {vram_free:.0f} MiB, Temp {gpu['temperature_c']}°C")

    # Metrics check from wandb or local
    sm = wdb.get("summary", {})
    gnorm = sm.get("gnorm")
    ema_loss = sm.get("ema_loss")
    val_acc = sm.get("val_acc")

    if gnorm is not None:
        if gnorm > 150:
            alerts.append(f"⚠️ [METRIC WARNING] Gradient norm high ({gnorm:.1f} > 150). Possible gradient explosion.")
        else:
            alerts.append(f"✅ [METRIC OK] Gradient norm stable ({gnorm:.1f}).")

    if ema_loss is not None:
        if ema_loss > 3.0:
            alerts.append(f"⚠️ [METRIC WARNING] Loss EMA high ({ema_loss:.3f}). Verify learning rate.")
        else:
            alerts.append(f"✅ [METRIC OK] Loss EMA is {ema_loss:.4f}.")

    if val_acc is not None:
        alerts.append(f"📊 [VAL ACCURACY] Overall: {val_acc * 100:.2f}% (Best: {sm.get('best_val_acc', 0) * 100:.2f}%)")

    return alerts


def main():
    print("=" * 60)
    print("        SystemOne Training Diagnostic & W&B Check       ")
    print("=" * 60)

    gpu = get_gpu_status()
    proc = get_running_process()
    local = get_latest_local_log_metrics()
    wdb = get_wandb_status()

    # 1. Diagnostic Summary
    alerts = evaluate_health(proc, gpu, wdb, local)
    print("\n[Health Diagnostic Assessment]")
    for a in alerts:
        print(f"  {a}")

    # 2. W&B Status
    print("\n[W&B Run Status]")
    if wdb.get("run_id"):
        print(f"  Run ID:    {wdb['run_id']} ({wdb['run_name']})")
        print(f"  State:     {wdb['state']}")
        print(f"  URL:       {wdb['url']}")
        sm = wdb.get("summary", {})
        print(f"  Optimizer Step: {sm.get('step')}")
        print(f"  Loss (Raw / EMA / Avg): {sm.get('raw_loss', 0):.4f} / {sm.get('ema_loss', 0):.4f} / {sm.get('avg_loss', 0):.4f}")
        print(f"  Current LR:     {sm.get('lr', 0):.3e}")
        print(f"  Gradient Norm:  {sm.get('gnorm', 0)}")

        gyms = wdb.get("gym_accuracies", {})
        if gyms:
            print("\n  Task-by-Task Val Accuracy:")
            for gym, acc in sorted(gyms.items()):
                bar = "█" * int(acc * 20)
                print(f"    - {gym:<22}: {acc * 100:5.1f}% | {bar}")
    else:
        print(f"  W&B Status: {wdb.get('status')} ({wdb.get('error', '')})")

    # 3. Local Log Steps
    if local.get("recent_steps"):
        print(f"\n[Latest Local Log Steps: {local['file']}]")
        for s in local["recent_steps"]:
            print(f"  step {s['step']}/{s['max_steps']} | loss={s['loss']:.4f} ema={s['ema']:.4f} lr={s['lr']:.2e} gnorm={s['gnorm']:.1f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
