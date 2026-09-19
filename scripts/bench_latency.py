#!/usr/bin/env python3
"""Latency / throughput smoke bench for systemone-lite vs published Jev numbers."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from systemone_lite.client import SystemOneClient, choice, noul, score
from systemone_lite.infer import reset_engine

ROOT = Path(__file__).resolve().parents[1]

# TypeSafe public claims (docs / launch materials). Not a live call.
JEV_PUBLISHED = {
    "source": "TypeSafe public claims (docs/blog; not measured on this machine)",
    "e2e_latency_ms": {"typical_low": 70, "typical_high": 500},
    "input_price_per_mtok_usd": 0.042,
    "notes": [
        "Measured from TypeSafe servers (often cited from US West Coast clients).",
        "Includes their parallel sampler; network RTT applies for remote callers.",
        "Output tokens billed as free / too cheap to meter.",
    ],
}


def make_questions(n: int) -> dict[str, Any]:
    """Build n independent questions mixing noul/choice/score."""
    templates = [
        lambda i: (
            f"flag_{i}",
            noul(f"Is signal #{i} present in the state?"),
        ),
        lambda i: (
            f"route_{i}",
            choice(
                f"Route bucket #{i}",
                {
                    "billing": "payment issues",
                    "technical": "bugs / outages",
                    "sales": "pricing / accounts",
                    "other": None,
                },
            ),
        ),
        lambda i: (
            f"score_{i}",
            score(f"Severity dimension #{i}", ["low", "medium", "high"]),
        ),
    ]
    out: dict[str, Any] = {}
    for i in range(n):
        key, q = templates[i % 3](i)
        out[key] = q
    return out


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def bench_case(
    client: SystemOneClient,
    *,
    name: str,
    state: str,
    n_questions: int,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    questions = make_questions(n_questions)
    for _ in range(warmup):
        client.system_one(state=state, questions=questions)

    times_ms: list[float] = []
    last_tokens = 0
    for _ in range(runs):
        t0 = time.perf_counter()
        resp = client.system_one(state=state, questions=questions)
        dt = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt)
        last_tokens = resp.usage.input_tokens

    times_ms.sort()
    mean = statistics.fmean(times_ms)
    return {
        "name": name,
        "n_questions": n_questions,
        "state_chars": len(state),
        "runs": runs,
        "warmup": warmup,
        "input_tokens_last": last_tokens,
        "latency_ms": {
            "min": round(times_ms[0], 2),
            "p50": round(percentile(times_ms, 50), 2),
            "p95": round(percentile(times_ms, 95), 2),
            "max": round(times_ms[-1], 2),
            "mean": round(mean, 2),
        },
        "throughput": {
            "requests_per_sec": round(1000.0 / mean, 2) if mean else None,
            "questions_per_sec": round((1000.0 / mean) * n_questions, 2) if mean else None,
        },
    }


def device_info() -> dict[str, Any]:
    info: dict[str, Any] = {"torch_cuda": False}
    try:
        import torch

        info["torch_cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
        info["device_used"] = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    return info


def compare_row(our_p50: float) -> dict[str, Any]:
    lo = JEV_PUBLISHED["e2e_latency_ms"]["typical_low"]
    hi = JEV_PUBLISHED["e2e_latency_ms"]["typical_high"]
    return {
        "our_p50_ms": our_p50,
        "jev_claimed_range_ms": [lo, hi],
        "vs_jev_low_x": round(our_p50 / lo, 2),
        "vs_jev_high_x": round(our_p50 / hi, 2),
        "interpretation": (
            "faster_than_jev_high"
            if our_p50 < hi
            else "slower_than_jev_high"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="systemone-lite latency bench")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmarks" / "latency_latest.json",
    )
    args = parser.parse_args()

    reset_engine()
    client = SystemOneClient(model="systemone-lite-latest")

    short_state = (
        "Help! My payouts have been failing for 3 days and I need this fixed ASAP."
    )
    long_state = (short_state + "\n") + ("Customer notes: still blocked. " * 200)

    cases = [
        ("short_1q", short_state, 1),
        ("short_3q", short_state, 3),
        ("short_8q", short_state, 8),
        ("short_13q", short_state, 13),
        ("long_3q", long_state, 3),
        ("long_13q", long_state, 13),
    ]

    results = []
    for name, state, nq in cases:
        print(f"bench {name} ...", flush=True)
        results.append(
            bench_case(
                client,
                name=name,
                state=state,
                n_questions=nq,
                warmup=args.warmup,
                runs=args.runs,
            )
        )

    short_3 = next(r for r in results if r["name"] == "short_3q")
    report = {
        "project": "systemone-lite",
        "model_alias": "systemone-lite-latest",
        "resolved_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "device": device_info(),
        "method": {
            "warmup": args.warmup,
            "runs": args.runs,
            "timer": "time.perf_counter around in-process system_one()",
            "includes": "tokenize + batched forward + assemble (no HTTP)",
        },
        "cases": results,
        "jev_published": JEV_PUBLISHED,
        "comparison_short_3q": compare_row(short_3["latency_ms"]["p50"]),
        "caveats": [
            "Jev numbers are vendor-published ranges, not live API calls from this host.",
            "No TYPESAFE_API_KEY available for head-to-head RTT measurement.",
            "Local bench excludes HTTP; add ~1–5ms locally or tens of ms over WAN for fair E2E.",
            "Accuracy is not measured here — only latency/throughput.",
            "GPU contention / first-load excluded after warmup.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
