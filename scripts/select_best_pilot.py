#!/usr/bin/env python3
"""Evaluate best_val across all lr-pilot runs and select the global winning checkpoint."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


def main():
    trials = [
        ("Trial-A (2e-6)", CHECKPOINTS_DIR / "lr-pilot-A-qwen-2e6"),
        ("Trial-B (5e-6)", CHECKPOINTS_DIR / "lr-pilot-B-qwen-5e6"),
        ("Trial-C (1e-5)", CHECKPOINTS_DIR / "lr-pilot-C-qwen-1e5"),
    ]

    results = []
    for name, trial_dir in trials:
        best_val_dir = trial_dir / "best_val"
        meta_json = best_val_dir / "best_val_meta.json"
        acc = None
        step = None
        if meta_json.exists():
            try:
                data = json.loads(meta_json.read_text())
                acc = float(data.get("val_accuracy"))
                step = int(data.get("step"))
            except Exception:
                pass

        # If files don't exist, check train_meta.json
        meta_file = trial_dir / "train_meta.json"
        if acc is None and meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                acc = meta.get("best_val_accuracy")
                step = meta.get("steps")
            except Exception:
                pass

        results.append({
            "name": name,
            "dir": str(trial_dir.relative_to(REPO_ROOT)),
            "best_val_dir": str(best_val_dir.relative_to(REPO_ROOT)) if best_val_dir.exists() else None,
            "best_val_accuracy": acc,
            "best_step": step,
        })

    # Sort by accuracy
    valid_results = [r for r in results if r["best_val_accuracy"] is not None]
    if not valid_results:
        print("No completed trial evaluations found yet.")
        return

    valid_results.sort(key=lambda x: x["best_val_accuracy"], reverse=True)
    winner = valid_results[0]

    # Create / update symlink checkpoints/best_pilot_model
    best_link = CHECKPOINTS_DIR / "best_pilot_model"
    if best_link.is_symlink() or best_link.exists():
        best_link.unlink()

    target_path = Path(winner["best_val_dir"]).resolve()
    best_link.symlink_to(target_path)

    report_path = REPO_ROOT / "benchmarks" / "pilot_selection_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "winning_trial": winner["name"],
        "winning_accuracy": winner["best_val_accuracy"],
        "winning_step": winner["best_step"],
        "winning_dir": winner["best_val_dir"],
        "all_trials": results,
    }
    report_path.write_text(json.dumps(report_data, indent=2) + "\n")

    print("=" * 60)
    print("      PILOT SELECTION REPORT (Auto-Winner Selection)      ")
    print("=" * 60)
    for r in valid_results:
        print(f"  {r['name']:<20}: Val Acc = {r['best_val_accuracy']*100:.2f}% (at step {r['best_step']})")
    print("-" * 60)
    print(f"WINNER: {winner['name']} with {winner['best_val_accuracy']*100:.2f}%")
    print(f"Linked checkpoints/best_pilot_model -> {winner['best_val_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
