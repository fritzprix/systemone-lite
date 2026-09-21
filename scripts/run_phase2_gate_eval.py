#!/usr/bin/env python3
"""Run Phase 2 exit-gate evals and write benchmarks/spatial_v2_report.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_eval(
    *,
    model: str,
    data: Path,
    out: Path,
    gyms: str = "",
    limit: int = 0,
    seed: int = 0,
) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "phase2_eval.py"),
        "--model",
        model,
        "--data",
        str(data),
        "--out",
        str(out),
        "--seed",
        str(seed),
    ]
    if gyms:
        cmd.extend(["--gyms", gyms])
    if limit:
        cmd.extend(["--limit", str(limit)])
    print("→", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    return json.loads(out.read_text(encoding="utf-8"))


def delta(a: float, b: float) -> float:
    return b - a


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 gate evaluation suite")
    parser.add_argument(
        "--spatial-model",
        default=str(ROOT / "checkpoints" / "systemone-spatial-v2"),
    )
    parser.add_argument(
        "--phase1-model",
        default=str(ROOT / "checkpoints" / "systemone-mixed-sft"),
    )
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmarks" / "spatial_v2_report.json",
    )
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bench = ROOT / "benchmarks"
    bench.mkdir(parents=True, exist_ok=True)
    scratch = bench / "_phase2_gate"
    scratch.mkdir(parents=True, exist_ok=True)

    spatial_gyms = "sokoban,game2048,gridworld,connect4"
    models = {
        "phase1_mixed": args.phase1_model,
        "spatial_v2": args.spatial_model,
    }
    if not args.skip_base:
        models = {"base": args.base_model, **models}

    results: dict[str, dict] = {}
    suites = {
        "spatial_heldout": (ROOT / "data" / "phase2_eval_4k.jsonl", spatial_gyms, 0),
        "invariance": (ROOT / "data" / "phase2_eval_invariance.jsonl", spatial_gyms, 0),
        "chess_move_2d": (ROOT / "data" / "chess_eval_5k_2d.jsonl", "chess", 500),
        "general_iid": (ROOT / "data" / "general_eval.jsonl", "", 0),
        "general_hard": (ROOT / "data" / "general_eval_hard.jsonl", "", 0),
    }

    for suite_name, (data, gyms, limit) in suites.items():
        results[suite_name] = {}
        for model_name, model_path in models.items():
            out = scratch / f"{suite_name}__{model_name}.json"
            results[suite_name][model_name] = run_eval(
                model=model_path,
                data=data,
                out=out,
                gyms=gyms,
                limit=limit,
                seed=args.seed,
            )

    def acc(suite: str, model: str) -> float:
        return float(results[suite][model]["accuracy"])

    spatial_v2 = results["spatial_heldout"]["spatial_v2"]
    phase1 = results["spatial_heldout"]["phase1_mixed"]
    per_gym_delta = {}
    for gym, stats in spatial_v2["per_gym"].items():
        p1 = phase1["per_gym"].get(gym, {}).get("accuracy", 0.0)
        per_gym_delta[gym] = {
            "phase1_mixed": p1,
            "spatial_v2": stats["accuracy"],
            "delta": delta(p1, stats["accuracy"]),
            "n": stats["n"],
        }

    chess_p1 = 0.236  # published Phase 1 shuffled 2D number
    chess_now = acc("chess_move_2d", "spatial_v2")
    chess_phase1_rerun = acc("chess_move_2d", "phase1_mixed")
    gen_iid_p1 = 0.7813888888888889
    gen_hard_p1 = 0.7333333333333333

    gates = {
        "spatial_lift_vs_phase1": {
            "pass": all(v["delta"] > 0.05 for v in per_gym_delta.values()),
            "per_gym_delta": per_gym_delta,
            "rule": "each spatial gym Δ > +0.05 vs Phase 1 mixed",
        },
        "chess_no_regress": {
            "pass": chess_now >= chess_p1 - 0.03,
            "spatial_v2": chess_now,
            "phase1_published": chess_p1,
            "phase1_rerun": chess_phase1_rerun,
            "tolerance": -0.03,
            "rule": "chess shuffled 2D ≥ published 0.24 − 0.03",
        },
        "general_no_catastrophic_forget": {
            "pass": (
                acc("general_iid", "spatial_v2") >= gen_iid_p1 - 0.05
                and acc("general_hard", "spatial_v2") >= gen_hard_p1 - 0.05
            ),
            "iid": {
                "spatial_v2": acc("general_iid", "spatial_v2"),
                "phase1_published": gen_iid_p1,
                "phase1_rerun": acc("general_iid", "phase1_mixed"),
            },
            "hard": {
                "spatial_v2": acc("general_hard", "spatial_v2"),
                "phase1_published": gen_hard_p1,
                "phase1_rerun": acc("general_hard", "phase1_mixed"),
            },
            "tolerance": -0.05,
            "rule": "general iid/hard within −0.05 of Phase 1 published",
        },
        "invariance_not_collapse": {
            "pass": (
                acc("invariance", "spatial_v2")
                >= acc("spatial_heldout", "spatial_v2") - 0.08
            ),
            "invariance": acc("invariance", "spatial_v2"),
            "canonical_spatial": acc("spatial_heldout", "spatial_v2"),
            "tolerance": -0.08,
            "rule": "invariance accuracy within −0.08 of spatial held-out",
        },
    }

    report = {
        "train": {
            "checkpoint": args.spatial_model,
            "phase1_checkpoint": args.phase1_model,
            "base_model": args.base_model,
        },
        "suites": {
            name: {
                model: {
                    "n": payload["n"],
                    "accuracy": payload["accuracy"],
                    "per_gym": payload.get("per_gym", {}),
                }
                for model, payload in suite.items()
            }
            for name, suite in results.items()
        },
        "gates": gates,
        "gate_pass": all(g["pass"] for g in gates.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate_pass": report["gate_pass"], "gates": {
        k: v["pass"] for k, v in gates.items()
    }}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
