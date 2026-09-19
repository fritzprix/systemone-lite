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


def test_heuristic_samples_have_labels() -> None:
    board = chess.Board()
    samples = build_samples_for_position(board, engine_path=None, include_stages=True)
    assert any(s.task == "move" for s in samples)
    move = next(s for s in samples if s.task == "move")
    assert move.label_alias in move.criteria
    best = heuristic_best_move(board)
    assert move.label_key == best.uci()
