"""Unit tests for chess distillation helpers (no Stockfish required)."""

from __future__ import annotations

import chess

from systemone_lite.chess_data import (
    alias_criteria,
    build_samples_for_position,
    heuristic_best_move,
)


def test_alias_single_letters() -> None:
    criteria, alias_to_key = alias_criteria({"e2e4": "push", "g1f3": "develop"})
    assert set(criteria) == {"A", "B"}
    assert alias_to_key["A"] == "e2e4"


def test_default_build_is_staged_only() -> None:
    board = chess.Board()
    samples = build_samples_for_position(board, engine_path=None)
    tasks = {s.task for s in samples}
    assert tasks == {"piece", "destination"}
    assert all(s.meta.get("task_schema") == "staged_v1" for s in samples)
    assert all(len(s.criteria) <= 8 for s in samples)
    best = heuristic_best_move(board)
    piece = next(s for s in samples if s.task == "piece")
    dest = next(s for s in samples if s.task == "destination")
    assert piece.label_key == chess.square_name(best.from_square)
    assert dest.label_key == best.uci()
    assert piece.label_alias in piece.criteria
    assert dest.label_alias in dest.criteria
    assert "board_2d_map" in piece.state
    assert "a b c d e f g h" in piece.state["board_2d_map"]
    assert "selected_piece" in dest.state


def test_option_caps_respected() -> None:
    board = chess.Board()
    samples = build_samples_for_position(
        board,
        engine_path=None,
        max_piece_options=4,
        max_dest_options=3,
    )
    piece = next(s for s in samples if s.task == "piece")
    dest = next(s for s in samples if s.task == "destination")
    assert len(piece.criteria) <= 4
    assert len(dest.criteria) <= 3
    assert piece.label_alias in piece.criteria
    assert dest.label_alias in dest.criteria


def test_include_move_is_capped() -> None:
    board = chess.Board()
    samples = build_samples_for_position(
        board,
        engine_path=None,
        include_move=True,
        include_stages=False,
        max_move_options=6,
    )
    assert len(samples) == 1
    move = samples[0]
    assert move.task == "move"
    assert len(move.criteria) <= 6
    best = heuristic_best_move(board)
    assert move.label_key == best.uci()
    assert move.label_alias in move.criteria


def test_move_label_alias_debias_with_include_move() -> None:
    board = chess.Board()
    aliases: set[str] = set()
    for _ in range(20):
        samples = build_samples_for_position(
            board,
            engine_path=None,
            include_move=True,
            include_stages=False,
            max_move_options=6,
        )
        aliases.add(samples[0].label_alias)
    assert len(aliases) > 1
