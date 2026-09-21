#!/usr/bin/env python3
"""Evaluate systemone-lite on JevBench datasets.

Supports both in-process engine inference (direct PyTorch KV-cache path)
and HTTP endpoint evaluation (POST /v1/systemone).

Usage examples:
  # In-process evaluation of fine-tuned checkpoint
  python scripts/jevbench_eval.py --model checkpoints/systemone-mixed-sft --splits easy,original

  # Evaluation against running HTTP server
  python scripts/jevbench_eval.py --endpoint http://127.0.0.1:8000 --splits easy

  # Full evaluation on all public splits
  python scripts/jevbench_eval.py --model checkpoints/systemone-mixed-sft --out benchmarks/jevbench_mixed_sft.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure project and third_party/jevbench are importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "jevbench"))

from jevbench.metrics import brier_score, ece_top_label
from jevbench.scoring import score_task
from jevbench.tasks import load_jsonl
from systemone_lite.schema import SystemOneRequest

logger = logging.getLogger("jevbench_eval")


def evaluate_inprocess(
    engine,
    task,
    *,
    temperature: float | None = None,
) -> tuple[dict[str, Any], float, str | None]:
    """Evaluate a single task using the in-process Engine."""
    q = {"type": task.question["type"], "instructions": task.question["instructions"]}
    if task.question.get("criteria") is not None:
        q["criteria"] = task.question["criteria"]

    req = SystemOneRequest(
        model=getattr(engine, "model_id", "systemone-lite-latest"),
        state=task.state,
        questions={"decision": q},
    )

    t0 = time.perf_counter()
    try:
        res = engine.decide(req, temperature=temperature)
        latency = (time.perf_counter() - t0) * 1000.0  # ms
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000.0
        return {}, latency, str(exc)

    ans = res.answers.get("decision")
    if ans is None:
        return {}, latency, "missing answers.decision"

    qtype = task.question["type"]
    if qtype == "noul":
        p_yes = float(getattr(ans, "noul", 0.0))
        probs = {"yes": p_yes, "no": round(1.0 - p_yes, 6)}
    elif qtype == "choice":
        probs = dict(getattr(ans, "probabilities", {}))
    elif qtype == "score":
        probs = dict(getattr(ans, "probabilities", {}))
    else:
        probs = {}

    return probs, latency, None


def evaluate_http(endpoint: str, model: str, task, timeout_s: float = 30.0) -> tuple[dict[str, Any], float, str | None]:
    """Evaluate a single task via HTTP POST /v1/systemone."""
    q = {"type": task.question["type"], "instructions": task.question["instructions"]}
    if task.question.get("criteria") is not None:
        q["criteria"] = task.question["criteria"]

    body = {
        "model": model,
        "state": task.state,
        "questions": {"decision": q},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/systemone",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
            latency = (time.perf_counter() - t0) * 1000.0
    except urllib.error.HTTPError as exc:
        latency = (time.perf_counter() - t0) * 1000.0
        err_msg = exc.read().decode("utf-8", errors="replace")
        return {}, latency, f"HTTP {exc.code}: {err_msg}"
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000.0
        return {}, latency, str(exc)

    ans = raw.get("answers", {}).get("decision")
    if not isinstance(ans, dict):
        return {}, latency, "missing answers.decision in response"

    qtype = task.question["type"]
    if qtype == "noul":
        p_yes = float(ans.get("noul", 0.0))
        probs = {"yes": p_yes, "no": round(1.0 - p_yes, 6)}
    elif qtype in ("choice", "score"):
        probs = ans.get("probabilities", {})
    else:
        probs = {}

    return probs, latency, None


def compute_percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    vals = sorted(values)
    n = len(vals)

    def p(pct: float) -> float:
        idx = min(int(n * pct), n - 1)
        return round(vals[idx], 2)

    return {
        "p50": p(0.50),
        "p90": p(0.90),
        "p95": p(0.95),
        "p99": p(0.99),
        "mean": round(sum(vals) / n, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on JevBench")
    parser.add_argument(
        "--model",
        default="checkpoints/systemone-mixed-sft",
        help="Model path or Hugging Face ID (for in-process inference)",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="HTTP endpoint URL (e.g. http://127.0.0.1:8000) for API evaluation",
    )
    parser.add_argument(
        "--splits",
        default="easy,original,hard",
        help="Comma-separated splits to run: easy, original, hard (or dataset path)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "third_party" / "jevbench" / "datasets" / "public",
        help="Directory containing public jsonl files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of items per split (0 = unlimited)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to save output JSON report",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-item results",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for option logits (T>1 softens / reduces overconfidence)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("systemone_lite.infer").setLevel(logging.ERROR)

    # 1. Load tasks
    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    tasks = []
    for s in split_names:
        file_path = Path(s)
        if not file_path.is_file():
            file_path = args.data_dir / f"{s}.jsonl"
        if not file_path.is_file():
            logger.error("Dataset file not found: %s", file_path)
            sys.exit(1)
        loaded = load_jsonl(str(file_path))
        if args.limit > 0:
            loaded = loaded[: args.limit]
        for t in loaded:
            t.split = s  # tag task with split name
        tasks.extend(loaded)

    logger.info("Loaded %d tasks across splits: %s", len(tasks), split_names)

    # 2. Setup engine or endpoint
    engine = None
    if not args.endpoint:
        logger.info(
            "Initializing in-process engine for model: %s (temperature=%s)",
            args.model,
            args.temperature,
        )
        from systemone_lite.infer import get_engine, reset_engine

        reset_engine()
        engine = get_engine(args.model, temperature=args.temperature)

    # 3. Warm-up
    if len(tasks) > 0:
        logger.info("Warming up inference engine...")
        if engine:
            evaluate_inprocess(engine, tasks[0], temperature=args.temperature)
        else:
            evaluate_http(args.endpoint, args.model, tasks[0])

    # 4. Evaluation loop
    results = []
    latencies = []
    pairs_for_ece = []
    brier_scores = []

    by_split = defaultdict(lambda: {"total": 0, "correct": 0, "valid": 0})
    by_type = defaultdict(lambda: {"total": 0, "correct": 0, "valid": 0})
    by_family = defaultdict(lambda: {"total": 0, "correct": 0, "valid": 0})

    logger.info("Running evaluation on %d tasks...", len(tasks))
    start_total_time = time.perf_counter()

    for idx, task in enumerate(tasks):
        if engine:
            probs, lat_ms, error = evaluate_inprocess(
                engine, task, temperature=args.temperature
            )
        else:
            probs, lat_ms, error = evaluate_http(args.endpoint, args.model, task)

        latencies.append(lat_ms)

        if error:
            scored = {
                "valid": False,
                "correct": False,
                "error": error,
                "predicted": None,
            }
        else:
            scored = score_task(probs, task)

        is_valid = bool(scored.get("valid", False))
        is_correct = bool(scored.get("correct", False))
        pred = scored.get("predicted")
        qtype = task.question["type"]

        # Track stats
        by_split[task.split]["total"] += 1
        by_type[qtype]["total"] += 1
        by_family[task.family]["total"] += 1

        if is_valid:
            by_split[task.split]["valid"] += 1
            by_type[qtype]["valid"] += 1
            by_family[task.family]["valid"] += 1

        if is_correct:
            by_split[task.split]["correct"] += 1
            by_type[qtype]["correct"] += 1
            by_family[task.family]["correct"] += 1

        # Calibration
        if is_valid and probs:
            top_conf = max(probs.values()) if probs else 0.0
            pairs_for_ece.append((top_conf, is_correct))
            try:
                b_score = brier_score(probs, str(task.expected), [str(l) for l in task.labels])
                brier_scores.append(b_score)
            except Exception:
                pass

        item_res = {
            "id": task.id,
            "split": task.split,
            "family": task.family,
            "type": qtype,
            "expected": task.expected,
            "predicted": pred,
            "correct": is_correct,
            "valid": is_valid,
            "latency_ms": round(lat_ms, 2),
            "error": scored.get("error"),
        }
        results.append(item_res)

        if args.verbose or (idx + 1) % 25 == 0 or (idx + 1) == len(tasks):
            acc_so_far = sum(1 for r in results if r["correct"]) / len(results) * 100.0
            print(
                f"[{idx+1:3d}/{len(tasks):3d}] "
                f"split={task.split:<8} type={qtype:<6} "
                f"acc={acc_so_far:5.1f}% lat={lat_ms:5.1f}ms "
                f"(id={task.id})"
            )

    total_wall_s = time.perf_counter() - start_total_time
    total_items = len(tasks)
    total_correct = sum(1 for r in results if r["correct"])
    total_valid = sum(1 for r in results if r["valid"])

    overall_acc = (total_correct / max(total_items, 1)) * 100.0
    valid_rate = (total_valid / max(total_items, 1)) * 100.0

    # Calibration metrics
    ece_data = ece_top_label(pairs_for_ece) if pairs_for_ece else {"ece": None}
    mean_brier = (sum(brier_scores) / len(brier_scores)) if brier_scores else None

    # Latency percentiles
    latency_stats = compute_percentiles(latencies)
    throughput = round(total_items / max(total_wall_s, 1e-4), 2)

    # Format per-breakdown results
    def format_breakdown(m: dict) -> dict:
        out = {}
        for k, v in sorted(m.items()):
            n = v["total"]
            c = v["correct"]
            val = v["valid"]
            out[k] = {
                "total": n,
                "valid": val,
                "correct": c,
                "accuracy": round(c / max(n, 1) * 100.0, 2),
                "valid_rate": round(val / max(n, 1) * 100.0, 2),
            }
        return out

    report = {
        "model": args.model if not args.endpoint else args.endpoint,
        "mode": "http" if args.endpoint else "in_process",
        "temperature": args.temperature,
        "total_items": total_items,
        "valid_items": total_valid,
        "correct_items": total_correct,
        "accuracy_pct": round(overall_acc, 2),
        "valid_rate_pct": round(valid_rate, 2),
        "ece": round(ece_data["ece"], 4) if ece_data.get("ece") is not None else None,
        "mean_brier": round(mean_brier, 4) if mean_brier is not None else None,
        "latency_ms": latency_stats,
        "throughput_decisions_per_sec": throughput,
        "total_wall_time_s": round(total_wall_s, 2),
        "splits": format_breakdown(by_split),
        "question_types": format_breakdown(by_type),
        "families": format_breakdown(by_family),
        "items": results,
    }

    # Print summary
    print("\n" + "=" * 65)
    print(f" JEVBENCH EVALUATION SUMMARY: {report['model']}")
    print("=" * 65)
    print(f"Temperature        : {args.temperature}")
    print(f"Total Decisions    : {total_items}")
    print(f"Accuracy           : {overall_acc:.2f}% ({total_correct}/{total_items})")
    print(f"Valid Rate         : {valid_rate:.2f}% ({total_valid}/{total_items})")
    if report["ece"] is not None:
        print(f"Calibration ECE    : {report['ece']:.4f}")
    if report["mean_brier"] is not None:
        print(f"Mean Brier Score   : {report['mean_brier']:.4f}")
    print(f"Latency (p50 / p90): {latency_stats['p50']:.1f} ms / {latency_stats['p90']:.1f} ms")
    print(f"Throughput         : {throughput:.1f} decisions/sec")
    print("-" * 65)
    print(f"{'Split / Type':<20} {'Total':>7} {'Correct':>9} {'Accuracy':>10}")
    print("-" * 65)
    print("[Splits]")
    for s, data in report["splits"].items():
        print(f"  {s:<18} {data['total']:>7} {data['correct']:>9} {data['accuracy']:>9.1f}%")
    print("[Question Types]")
    for qt, data in report["question_types"].items():
        print(f"  {qt:<18} {data['total']:>7} {data['correct']:>9} {data['accuracy']:>9.1f}%")
    print("=" * 65)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Report saved to %s", args.out)


if __name__ == "__main__":
    main()
