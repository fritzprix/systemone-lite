#!/usr/bin/env python3
"""Latency bench: option-scoring System One vs autoregressive JSON generation."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from systemone_lite.client import SystemOneClient, choice, noul, score
from systemone_lite.infer import get_engine, reset_engine
from systemone_lite.prompt import render_state

ROOT = Path(__file__).resolve().parents[1]

JEV_PUBLISHED = {
    "source": "TypeSafe public claims (docs/blog; not measured on this machine)",
    "e2e_latency_ms": {"typical_low": 70, "typical_high": 500},
    "notes": [
        "Vendor-published cloud E2E range; not a live call from this host.",
    ],
}


def make_questions(n: int) -> dict[str, Any]:
    templates = [
        lambda i: (f"flag_{i}", noul(f"Is signal #{i} present in the state?")),
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


def summarize_times(times_ms: list[float]) -> dict[str, float]:
    times_ms = sorted(times_ms)
    mean = statistics.fmean(times_ms)
    return {
        "min": round(times_ms[0], 2),
        "p50": round(percentile(times_ms, 50), 2),
        "p95": round(percentile(times_ms, 95), 2),
        "max": round(times_ms[-1], 2),
        "mean": round(mean, 2),
    }


def device_info() -> dict[str, Any]:
    info: dict[str, Any] = {"torch_cuda": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
        info["device_used"] = "cuda"
    else:
        info["device_used"] = "cpu"
    return info


def build_ar_json_prompt(state: str, questions: dict[str, Any]) -> str:
    """Prompt for naive autoregressive structured JSON (full vocab generation)."""
    schema_lines: list[str] = []
    for qid, q in questions.items():
        if q["type"] == "noul":
            schema_lines.append(f'  "{qid}": "yes"|"no"')
        elif q["type"] == "choice":
            opts = " | ".join(json.dumps(k) for k in q["criteria"])
            schema_lines.append(f'  "{qid}": {opts}')
        else:
            levels = " | ".join(str(i) for i in range(len(q["criteria"])))
            schema_lines.append(f'  "{qid}": {levels}')
    schema = "{\n" + ",\n".join(schema_lines) + "\n}"
    return (
        f"### State\n{render_state(state)}\n\n"
        "### Task\n"
        "Output a single JSON object answering every field. "
        "Use only the allowed values. No markdown, no commentary.\n\n"
        f"### Schema\n{schema}\n\n"
        "### Answer\n"
    )


def ar_max_new_tokens(n_questions: int) -> int:
    # Rough budget: JSON braces + keys + values; leave headroom.
    return min(512, 40 + n_questions * 28)


def bench_option_scoring(
    client: SystemOneClient,
    *,
    state: str,
    questions: dict[str, Any],
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        client.system_one(state=state, questions=questions)

    times_ms: list[float] = []
    last_in = 0
    last_out = 0
    for _ in range(runs):
        t0 = time.perf_counter()
        resp = client.system_one(state=state, questions=questions)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        last_in = resp.usage.input_tokens
        last_out = resp.usage.output_tokens

    return {
        "mode": "option_scoring",
        "description": (
            "Batched next-token logits; softmax over option token ids only; "
            "prefix KV when n_questions>=2"
        ),
        "latency_ms": summarize_times(times_ms),
        "input_tokens_last": last_in,
        "output_tokens_last": last_out,
    }


@torch.inference_mode()
def bench_ar_json(
    *,
    state: str,
    questions: dict[str, Any],
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    engine = get_engine("systemone-lite-latest")
    loaded = engine._loaded  # noqa: SLF001 — bench uses same loaded weights
    tokenizer = loaded.tokenizer
    model = loaded.model
    device = loaded.device

    prompt = build_ar_json_prompt(state, questions)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    enc = {k: v.to(device) for k, v in enc.items()}
    max_new = ar_max_new_tokens(len(questions))
    gen_kwargs = {
        "max_new_tokens": max_new,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    for _ in range(warmup):
        model.generate(**enc, **gen_kwargs)

    times_ms: list[float] = []
    last_in = int(enc["attention_mask"].sum().item())
    last_out = 0
    for _ in range(runs):
        t0 = time.perf_counter()
        out = model.generate(**enc, **gen_kwargs)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        last_out = int(out.shape[-1] - enc["input_ids"].shape[-1])

    return {
        "mode": "ar_json",
        "description": (
            "Autoregressive generate() of a multi-field JSON object "
            f"(greedy, max_new_tokens={max_new}, full vocabulary)"
        ),
        "latency_ms": summarize_times(times_ms),
        "input_tokens_last": last_in,
        "output_tokens_last": last_out,
        "max_new_tokens": max_new,
    }


def speedup(ar_p50: float, opt_p50: float) -> float | None:
    if opt_p50 <= 0:
        return None
    return round(ar_p50 / opt_p50, 2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare option-scoring vs AR JSON latency"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmarks" / "latency_vs_ar.json",
    )
    parser.add_argument(
        "--cases",
        default="short_3q,short_13q,long_3q,long_13q",
        help="Comma-separated case names to run",
    )
    args = parser.parse_args()

    reset_engine()
    client = SystemOneClient(model="systemone-lite-latest")

    short_state = (
        "Help! My payouts have been failing for 3 days and I need this fixed ASAP."
    )
    long_state = (short_state + "\n") + ("Customer notes: still blocked. " * 200)

    catalog = {
        "short_1q": (short_state, 1),
        "short_3q": (short_state, 3),
        "short_8q": (short_state, 8),
        "short_13q": (short_state, 13),
        "long_3q": (long_state, 3),
        "long_13q": (long_state, 13),
    }
    selected = [c.strip() for c in args.cases.split(",") if c.strip()]
    for name in selected:
        if name not in catalog:
            raise SystemExit(f"unknown case {name!r}; choose from {sorted(catalog)}")

    cases_out: list[dict[str, Any]] = []
    for name in selected:
        state, nq = catalog[name]
        questions = make_questions(nq)
        print(f"bench {name} option_scoring ...", flush=True)
        opt = bench_option_scoring(
            client,
            state=state,
            questions=questions,
            warmup=args.warmup,
            runs=args.runs,
        )
        print(f"bench {name} ar_json ...", flush=True)
        ar = bench_ar_json(
            state=state,
            questions=questions,
            warmup=args.warmup,
            runs=args.runs,
        )
        ar_p50 = ar["latency_ms"]["p50"]
        opt_p50 = opt["latency_ms"]["p50"]
        row = {
            "name": name,
            "n_questions": nq,
            "state_chars": len(state),
            "option_scoring": opt,
            "ar_json": ar,
            "speedup_ar_over_option_p50": speedup(ar_p50, opt_p50),
        }
        cases_out.append(row)
        print(
            f"  {name}: option p50={opt_p50} ms | ar p50={ar_p50} ms | "
            f"ratio={row['speedup_ar_over_option_p50']}x",
            flush=True,
        )

    report = {
        "project": "systemone-lite",
        "resolved_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "device": device_info(),
        "method": {
            "warmup": args.warmup,
            "runs": args.runs,
            "timer": "time.perf_counter (in-process, no HTTP)",
            "option_scoring": "SystemOneClient.system_one (batched option logits)",
            "ar_json": "model.generate greedy JSON for all fields (full vocab)",
        },
        "cases": cases_out,
        "jev_published": JEV_PUBLISHED,
        "caveats": [
            "AR path measures generate() wall time, not JSON parse success rate.",
            "AR max_new_tokens is a fixed budget; early EOS can shorten some runs.",
            "Option-scoring always returns schema-valid symbols; AR may emit invalid JSON.",
            "Not compared against TypeSafe/Jev hardware or sampler.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
