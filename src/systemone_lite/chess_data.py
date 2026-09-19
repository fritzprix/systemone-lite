"""Chess helpers for System One distillation / demos."""

from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def find_stockfish() -> str | None:
    env = os.environ.get("STOCKFISH_PATH")
    if env and Path(env).exists():
        return env
    found = shutil.which("stockfish")
    if found:
        return found
    # Debian/Ubuntu often install to /usr/games which may be off PATH.
    for candidate in (
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
    ):
        if Path(candidate).exists():
            return candidate
    repo_root = Path(__file__).resolve().parents[2]
    local = repo_root / "third_party" / "stockfish" / "stockfish-ubuntu-x86-64-avx2"
    if local.exists():
        return str(local)
    return None


def board_state(board: chess.Board) -> dict[str, Any]:
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "fullmove": board.fullmove_number,
        "legal_move_count": board.legal_moves.count(),
        "in_check": board.is_check(),
    }


def describe_piece(board: chess.Board, square_name: str) -> str:
    square = chess.parse_square(square_name)
    piece = board.piece_at(square)
    if piece is None:
        return square_name
    color = "white" if piece.color == chess.WHITE else "black"
    return f"{color} {chess.piece_name(piece.piece_type)} on {square_name}"


def describe_move(board: chess.Board, move: chess.Move) -> str:
    to_sq = chess.square_name(move.to_square)
    note = f"to {to_sq}"
    captured = board.piece_at(move.to_square)
    if captured:
        note += f", capture {chess.piece_name(captured.piece_type)}"
    if move.promotion:
        note += f", promote to {chess.piece_name(move.promotion)}"
    if board.is_castling(move):
        note += ", castle"
    if board.gives_check(move):
        note += ", check"
    return note


def legal_by_origin(board: chess.Board) -> dict[str, list[chess.Move]]:
    grouped: dict[str, list[chess.Move]] = {}
    for move in board.legal_moves:
        origin = chess.square_name(move.from_square)
        grouped.setdefault(origin, []).append(move)
    return grouped


def alias_criteria(options: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Map keys → single-token aliases A,B,C… and reverse map."""
    alias_to_key: dict[str, str] = {}
    criteria: dict[str, str] = {}
    for i, (key, desc) in enumerate(options.items()):
        alias = chr(ord("A") + i) if i < 26 else str(i)
        alias_to_key[alias] = key
        criteria[alias] = f"{key}: {desc}"
    return criteria, alias_to_key


def heuristic_score_move(board: chess.Board, move: chess.Move) -> float:
    """Cheap tactical prior when Stockfish is unavailable."""
    score = 0.0
    mover = board.piece_at(move.from_square)
    if mover is None:
        return -1e9
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        v = PIECE_VALUE.get(victim.piece_type, 0) if victim else 0
        # en passant
        if victim is None and board.is_en_passant(move):
            v = PIECE_VALUE[chess.PAWN]
        score += 10.0 * v - PIECE_VALUE[mover.piece_type]
    if board.gives_check(move):
        score += 45.0
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    center = {(3, 3), (3, 4), (4, 3), (4, 4)}
    if (to_file, to_rank) in center:
        score += 15.0
    # slight development bonus for knights/bishops leaving back rank
    if mover.piece_type in (chess.KNIGHT, chess.BISHOP):
        from_rank = chess.square_rank(move.from_square)
        back = 0 if mover.color == chess.WHITE else 7
        if from_rank == back:
            score += 12.0
    if board.is_castling(move):
        score += 40.0
    return score


def heuristic_best_move(board: chess.Board) -> chess.Move:
    moves = list(board.legal_moves)
    if not moves:
        raise ValueError("no legal moves")
    return max(moves, key=lambda m: (heuristic_score_move(board, m), m.uci()))


def stockfish_best_move(
    board: chess.Board,
    *,
    engine_path: str,
    movetime_ms: int = 50,
    engine: chess.engine.SimpleEngine | None = None,
) -> chess.Move:
    limit = chess.engine.Limit(time=movetime_ms / 1000.0)
    if engine is not None:
        result = engine.play(board, limit)
    else:
        with chess.engine.SimpleEngine.popen_uci(engine_path) as eng:
            result = eng.play(board, limit)
    if result.move is None:
        raise RuntimeError("Stockfish returned no move")
    return result.move


def label_best_move(
    board: chess.Board,
    *,
    engine_path: str | None,
    movetime_ms: int = 50,
    engine: chess.engine.SimpleEngine | None = None,
) -> tuple[chess.Move, str]:
    if engine_path:
        return stockfish_best_move(
            board,
            engine_path=engine_path,
            movetime_ms=movetime_ms,
            engine=engine,
        ), "stockfish"
    return heuristic_best_move(board), "heuristic"


@dataclass
class DistillSample:
    """One supervised System One choice example."""

    task: str  # "move" | "piece" | "destination"
    state: dict[str, Any]
    instructions: str
    criteria: dict[str, str]  # alias -> description
    label_alias: str
    label_key: str  # original square or uci
    meta: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "state": self.state,
            "instructions": self.instructions,
            "criteria": self.criteria,
            "label_alias": self.label_alias,
            "label_key": self.label_key,
            "meta": self.meta,
        }


MOVE_INSTRUCTIONS = (
    "Choose the best legal chess move for the side to move. "
    "Prefer development, center control, tactics, and king safety. "
    "Reply with exactly one option letter from Criteria."
)

PIECE_INSTRUCTIONS = (
    "Pick one piece that should move this turn. Prefer development, "
    "center control, and king safety. Reply with the option letter."
)

DEST_INSTRUCTIONS = (
    "A piece is already selected. Choose the best legal destination. "
    "Reply with the option letter."
)


def build_samples_for_position(
    board: chess.Board,
    *,
    engine_path: str | None,
    movetime_ms: int = 50,
    include_stages: bool = True,
    engine: chess.engine.SimpleEngine | None = None,
) -> list[DistillSample]:
    moves = list(board.legal_moves)
    if not moves:
        return []

    best, source = label_best_move(
        board,
        engine_path=engine_path,
        movetime_ms=movetime_ms,
        engine=engine,
    )
    state = board_state(board)
    samples: list[DistillSample] = []

    # Cap move list to 26 aliases.
    if len(moves) > 26:
        # Keep best + random subset of others.
        others = [m for m in moves if m != best]
        random.shuffle(others)
        moves = [best] + others[:25]

    move_opts = {m.uci(): describe_move(board, m) for m in moves}
    criteria, alias_to_key = alias_criteria(move_opts)
    key_to_alias = {v: k for k, v in alias_to_key.items()}
    if best.uci() not in key_to_alias:
        return samples
    samples.append(
        DistillSample(
            task="move",
            state=state,
            instructions=MOVE_INSTRUCTIONS,
            criteria=criteria,
            label_alias=key_to_alias[best.uci()],
            label_key=best.uci(),
            meta={"label_source": source, "n_options": len(criteria)},
        )
    )

    if not include_stages:
        return samples

    origin = chess.square_name(best.from_square)
    grouped = legal_by_origin(board)
    origin_opts = {sq: describe_piece(board, sq) for sq in sorted(grouped)}
    o_crit, o_alias = alias_criteria(origin_opts)
    o_key_to_alias = {v: k for k, v in o_alias.items()}
    if origin in o_key_to_alias:
        samples.append(
            DistillSample(
                task="piece",
                state=state,
                instructions=PIECE_INSTRUCTIONS,
                criteria=o_crit,
                label_alias=o_key_to_alias[origin],
                label_key=origin,
                meta={"label_source": source, "best_uci": best.uci()},
            )
        )

    dest_moves = grouped[origin]
    dest_opts = {m.uci(): describe_move(board, m) for m in dest_moves}
    d_crit, d_alias = alias_criteria(dest_opts)
    d_key_to_alias = {v: k for k, v in d_alias.items()}
    if best.uci() in d_key_to_alias:
        samples.append(
            DistillSample(
                task="destination",
                state={**state, "selected_piece": origin},
                instructions=(
                    f"The piece on {origin} is selected. "
                    + DEST_INSTRUCTIONS
                ),
                criteria=d_crit,
                label_alias=d_key_to_alias[best.uci()],
                label_key=best.uci(),
                meta={"label_source": source, "selected_piece": origin},
            )
        )

    return samples


def generate_positions(
    *,
    n_positions: int,
    max_plies: int = 40,
    seed: int = 0,
    engine_path: str | None = None,
    movetime_ms: int = 30,
) -> list[chess.Board]:
    """Self-play-ish positions labeled by engine/heuristic along the way."""
    rng = random.Random(seed)
    boards: list[chess.Board] = []
    engine: chess.engine.SimpleEngine | None = None
    if engine_path:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    try:
        while len(boards) < n_positions:
            board = chess.Board()
            plies = rng.randint(0, max_plies)
            for _ in range(plies):
                if board.is_game_over():
                    break
                if rng.random() < 0.75:
                    move, _ = label_best_move(
                        board,
                        engine_path=engine_path,
                        movetime_ms=movetime_ms,
                        engine=engine,
                    )
                else:
                    move = rng.choice(list(board.legal_moves))
                board.push(move)
                if len(boards) < n_positions and not board.is_game_over():
                    if rng.random() < 0.45:
                        boards.append(board.copy(stack=False))
            if not board.is_game_over() and len(boards) < n_positions:
                boards.append(board.copy(stack=False))
    finally:
        if engine is not None:
            engine.quit()

    return boards[:n_positions]
