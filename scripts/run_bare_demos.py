#!/usr/bin/env python3
"""Regenerate bare-face Base vs SFT demos (no keyword hints) and write a JSON report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import chess_demo  # noqa: E402
import chess_multistep_demo  # noqa: E402
import connect4_demo  # noqa: E402
import game2048_demo  # noqa: E402
import gridworld_demo  # noqa: E402
import sokoban_demo  # noqa: E402
from systemone_lite import SystemOneClient  # noqa: E402
from systemone_lite.infer import reset_engine  # noqa: E402

OUT = ROOT / "benchmarks" / "demos"
MODELS = {
    "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "sft_model": str(ROOT / "checkpoints" / "systemone-mixed-sft"),  # Phase 1
    "spatial_v2": str(ROOT / "checkpoints" / "systemone-spatial-v2"),  # Phase 2 gate
    "spatial_v2b": str(ROOT / "checkpoints" / "systemone-spatial-v2b"),  # continual 20k
}


def _client(model: str) -> SystemOneClient:
    reset_engine()
    return SystemOneClient(model=model)


def run_one(tag: str, model: str) -> dict:
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    client = _client(model)
    results: dict = {"model": model, "tag": tag}

    print(f"\n=== {tag}: chess_proper ===", flush=True)
    chess = chess_demo.play_game(
        client=client,
        max_plies=10,
        delay=0.0,
        animate=False,
        gif_path=out_dir / "chess_proper.gif",
        fps=3,
    )
    results["chess_proper"] = {
        "plies": chess["plies"],
        "moves": chess["history"],
        "avg_latency_ms": round(chess["avg_latency_ms"], 1),
        "result": chess["result"],
    }
    print(f"  moves={' '.join(chess['history'])}  avg={chess['avg_latency_ms']:.1f}ms", flush=True)

    print(f"=== {tag}: chess_multistep ===", flush=True)
    multi = chess_multistep_demo.play_multistep_game(
        client=client,
        max_plies=10,
        delay=0.0,
        step_delay=0.0,
        animate=False,
        gif_path=out_dir / "chess_multistep.gif",
        fps=3,
    )
    results["chess_multistep"] = {
        "plies": multi["plies"],
        "moves": multi["history"],
        "avg_latency_ms": round(multi["avg_latency_ms"], 1),
        "result": multi.get("result"),
    }
    print(f"  moves={' '.join(multi['history'])}  avg={multi['avg_latency_ms']:.1f}ms", flush=True)

    print(f"=== {tag}: game2048 ===", flush=True)
    g2048 = game2048_demo.play_2048_demo(
        client=client,
        max_steps=15,
        delay=0.0,
        animate=False,
        gif_path=out_dir / "game2048.gif",
        fps=3,
    )
    results["game2048"] = {
        "steps": g2048["steps"],
        "score": g2048["score"],
        "max_tile": g2048["max_tile"],
        "avg_latency_ms": round(g2048["avg_latency_ms"], 1),
    }
    print(
        f"  score={g2048['score']} max={g2048['max_tile']} avg={g2048['avg_latency_ms']:.1f}ms",
        flush=True,
    )

    print(f"=== {tag}: gridworld ===", flush=True)
    gw = gridworld_demo.play_gridworld_demo(
        client=client,
        max_steps=10,
        delay=0.0,
        animate=False,
        gif_path=out_dir / "gridworld.gif",
        fps=3,
    )
    results["gridworld"] = {
        "steps": gw["steps"],
        "won": gw["won"],
        "died": gw["died"],
        "avg_latency_ms": round(gw["avg_latency_ms"], 1),
    }
    print(f"  won={gw['won']} died={gw['died']}", flush=True)

    print(f"=== {tag}: sokoban ===", flush=True)
    sok = sokoban_demo.play_sokoban_demo(
        client=client,
        max_steps=10,
        delay=0.0,
        animate=False,
        gif_path=out_dir / "sokoban.gif",
        fps=3,
    )
    results["sokoban"] = {
        "steps": sok["steps"],
        "won": sok["won"],
        "avg_latency_ms": round(sok["avg_latency_ms"], 1),
    }
    print(f"  won={sok['won']} steps={sok['steps']}", flush=True)

    print(f"=== {tag}: connect4 ===", flush=True)
    c4 = connect4_demo.play_connect4_demo(
        client=client,
        max_turns=12,
        delay=0.0,
        animate=False,
        gif_path=out_dir / "connect4.gif",
        fps=3,
    )
    results["connect4"] = {
        "turns": c4["turns"],
        "winner": c4["winner"],
        "avg_latency_ms": round(c4["avg_latency_ms"], 1),
    }
    print(f"  turns={c4['turns']} winner={c4['winner']}", flush=True)

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate bare-face demos")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated MODEL tags to run (default: all)",
    )
    args = parser.parse_args()
    selected = (
        {t.strip() for t in args.only.split(",") if t.strip()}
        if args.only
        else set(MODELS)
    )
    unknown = selected - set(MODELS)
    if unknown:
        raise SystemExit(f"unknown --only tags: {sorted(unknown)}")

    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "bare_face_report.json"
    report = {
        "protocol": {
            "hints": "removed (no CAPTURES/CHECK/develop/RECOMMENDED/BLOCKED/DEADLY tags in option text)",
            "chess_options": "all legal moves up to 26 (FEN-seeded subsample); UCI-sorted aliases",
            "solver_override": "disabled (model choice used; illegal → first legal only)",
            "self_play": "both colors controlled by the same model",
        },
        "runs": {},
    }
    if report_path.exists() and args.only:
        try:
            prev = json.loads(report_path.read_text(encoding="utf-8"))
            report["runs"] = dict(prev.get("runs") or {})
        except json.JSONDecodeError:
            pass

    for tag, model in MODELS.items():
        if tag not in selected:
            continue
        report["runs"][tag] = run_one(tag, model)

    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
