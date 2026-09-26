#!/usr/bin/env python3
"""Build a chess distillation JSONL for System One SFT.

Default schema (staged_v1): piece + destination only, with option fan-out caps.
Uncapped full-legal ``move`` rows are not emitted unless ``--include-move``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import chess

from systemone_lite.chess_data import (
    build_samples_for_position,
    find_stockfish,
    generate_positions,
)

ROOT = Path(__file__).resolve().parents[1]


def refresh_dataset(
    input_path: Path,
    output_path: Path,
    *,
    seed: int = 42,
    include_move: bool = False,
    include_stages: bool = True,
    max_move_options: int = 6,
    max_piece_options: int = 8,
    max_dest_options: int = 8,
) -> int:
    """Refresh an existing chess JSONL into staged_v1 piece+destination samples.

    Source ``move`` rows are expanded to piece+destination from the same FEN
    and gold UCI. ``piece`` / ``destination`` rows are rebuilt from FEN + label.
    Move-only output is omitted unless ``include_move`` is set.
    """
    rng = random.Random(seed)
    updated = 0
    temp_out = output_path.with_suffix(".tmp")
    with input_path.open("r", encoding="utf-8") as fin, temp_out.open(
        "w", encoding="utf-8"
    ) as fout:
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
            elif task == "destination":
                best = chess.Move.from_uci(label_key)
            elif task == "piece":
                # Gold is an origin square; recover best as any legal move from
                # that square preferring meta best_uci when present.
                meta_uci = (row.get("meta") or {}).get("best_uci")
                if meta_uci:
                    best = chess.Move.from_uci(meta_uci)
                else:
                    origin_sq = chess.parse_square(label_key)
                    from_origin = [
                        m
                        for m in board.legal_moves
                        if m.from_square == origin_sq
                    ]
                    if not from_origin:
                        continue
                    best = from_origin[0]
            else:
                continue

            if best not in board.legal_moves:
                continue

            samples = build_samples_for_position(
                board,
                best_move=best,
                source=source,
                include_stages=include_stages,
                include_move=include_move,
                max_move_options=max_move_options,
                max_piece_options=max_piece_options,
                max_dest_options=max_dest_options,
            )
            for sample in samples:
                fout.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")
                updated += 1

    temp_out.replace(output_path)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill chess → System One JSONL (staged piece+destination)"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "chess_distill.jsonl")
    parser.add_argument(
        "--refresh-file",
        type=Path,
        default=None,
        help="Re-encode existing JSONL as staged_v1 (move rows expand to stages)",
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
    parser.add_argument(
        "--include-move",
        action="store_true",
        help="Also emit capped move samples (default: stages only)",
    )
    parser.add_argument(
        "--no-stages",
        action="store_true",
        help="Omit piece/destination (requires --include-move)",
    )
    parser.add_argument("--max-move-options", type=int, default=6)
    parser.add_argument("--max-piece-options", type=int, default=8)
    parser.add_argument("--max-dest-options", type=int, default=8)
    args = parser.parse_args()

    include_stages = not args.no_stages
    if args.no_stages and not args.include_move:
        parser.error("--no-stages requires --include-move")

    if args.refresh_file:
        out_path = (
            args.out
            if args.out != (ROOT / "data" / "chess_distill.jsonl")
            else args.refresh_file
        )
        n = refresh_dataset(
            args.refresh_file,
            out_path,
            seed=args.seed,
            include_move=args.include_move,
            include_stages=include_stages,
            max_move_options=args.max_move_options,
            max_piece_options=args.max_piece_options,
            max_dest_options=args.max_dest_options,
        )
        print(
            f"refreshed {n} staged samples → {out_path} "
            f"(include_move={args.include_move}, stages={include_stages})"
        )
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
                    include_stages=include_stages,
                    include_move=args.include_move,
                    max_move_options=args.max_move_options,
                    max_piece_options=args.max_piece_options,
                    max_dest_options=args.max_dest_options,
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
