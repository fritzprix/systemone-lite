#!/usr/bin/env python3
"""Build a chess distillation JSONL for System One SFT."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import chess

from systemone_lite.chess_data import (
    DEST_INSTRUCTIONS,
    PIECE_INSTRUCTIONS,
    DistillSample,
    alias_criteria,
    board_state,
    build_samples_for_position,
    describe_move,
    describe_piece,
    find_stockfish,
    generate_positions,
    legal_by_origin,
)

ROOT = Path(__file__).resolve().parents[1]


def refresh_dataset(input_path: Path, output_path: Path, seed: int = 42) -> int:
    """Refresh an existing chess JSONL with 2D board maps and debiased option shuffling."""
    rng = random.Random(seed)
    updated = 0
    temp_out = output_path.with_suffix(".tmp")
    with input_path.open("r", encoding="utf-8") as fin, temp_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task = row["task"]
            fen = row["state"]["fen"]
            board = chess.Board(fen)
            label_key = row["label_key"]
            source = row.get("meta", {}).get("label_source", "stockfish")

            random.seed(rng.randint(0, 10**8))

            if task == "move":
                best = chess.Move.from_uci(label_key)
                samples = build_samples_for_position(
                    board,
                    best_move=best,
                    source=source,
                    include_stages=False,
                )
                if not samples:
                    continue
                fout.write(json.dumps(samples[0].to_json(), ensure_ascii=False) + "\n")
            elif task == "piece":
                origin = label_key
                grouped = legal_by_origin(board)
                origin_keys = list(grouped.keys())
                random.shuffle(origin_keys)
                origin_opts = {sq: describe_piece(board, sq) for sq in origin_keys}
                o_crit, o_alias = alias_criteria(origin_opts)
                o_key_to_alias = {v: k for k, v in o_alias.items()}
                sample = DistillSample(
                    task="piece",
                    state=board_state(board),
                    instructions=PIECE_INSTRUCTIONS,
                    criteria=o_crit,
                    label_alias=o_key_to_alias.get(origin, "A"),
                    label_key=origin,
                    meta={"label_source": source, "gym": "chess"},
                )
                fout.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")
            elif task == "destination":
                best = chess.Move.from_uci(label_key)
                origin = chess.square_name(best.from_square)
                grouped = legal_by_origin(board)
                dest_moves = list(grouped.get(origin, []))
                random.shuffle(dest_moves)
                dest_opts = {m.uci(): describe_move(board, m) for m in dest_moves}
                d_crit, d_alias = alias_criteria(dest_opts)
                d_key_to_alias = {v: k for k, v in d_alias.items()}
                sample = DistillSample(
                    task="destination",
                    state={**board_state(board), "selected_piece": origin},
                    instructions=f"The piece on {origin} is selected. {DEST_INSTRUCTIONS}",
                    criteria=d_crit,
                    label_alias=d_key_to_alias.get(best.uci(), "A"),
                    label_key=best.uci(),
                    meta={
                        "label_source": source,
                        "selected_piece": origin,
                        "gym": "chess",
                    },
                )
                fout.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")
            updated += 1

    temp_out.replace(output_path)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill chess → System One JSONL")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "chess_distill.jsonl")
    parser.add_argument(
        "--refresh-file",
        type=Path,
        default=None,
        help="Re-encode existing JSONL with 2D board maps and debiased option shuffling",
    )
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

    if args.refresh_file:
        out_path = args.out if args.out != (ROOT / "data" / "chess_distill.jsonl") else args.refresh_file
        n = refresh_dataset(args.refresh_file, out_path, seed=args.seed)
        print(f"refreshed {n} samples with 2D board maps and debiased labels → {out_path}")
        return

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
