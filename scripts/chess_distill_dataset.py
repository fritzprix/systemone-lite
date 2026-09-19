#!/usr/bin/env python3
"""Build a chess distillation JSONL for System One SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from systemone_lite.chess_data import (
    build_samples_for_position,
    find_stockfish,
    generate_positions,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill chess → System One JSONL")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "chess_distill.jsonl")
    parser.add_argument("--positions", type=int, default=500)
    parser.add_argument("--max-plies", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--movetime-ms", type=int, default=40)
    parser.add_argument(
        "--stockfish",
        type=str,
        default=None,
        help="Path to stockfish (default: STOCKFISH_PATH or PATH)",
    )
    parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Force heuristic labels even if Stockfish exists",
    )
    parser.add_argument("--no-stages", action="store_true", help="Only full-move samples")
    args = parser.parse_args()

    engine = None if args.heuristic_only else (args.stockfish or find_stockfish())
    print(f"labeler: {'stockfish:' + engine if engine else 'heuristic'}")

    boards = generate_positions(
        n_positions=args.positions,
        max_plies=args.max_plies,
        seed=args.seed,
        engine_path=engine,
        movetime_ms=args.movetime_ms,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    engine_proc = None
    if engine:
        import chess.engine

        engine_proc = chess.engine.SimpleEngine.popen_uci(engine)
    try:
        with args.out.open("w", encoding="utf-8") as fh:
            for board in boards:
                for sample in build_samples_for_position(
                    board,
                    engine_path=engine,
                    movetime_ms=args.movetime_ms,
                    include_stages=not args.no_stages,
                    engine=engine_proc,
                ):
                    fh.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")
                    n += 1
    finally:
        if engine_proc is not None:
            engine_proc.quit()

    print(f"wrote {n} samples from {len(boards)} positions → {args.out}")


if __name__ == "__main__":
    main()
