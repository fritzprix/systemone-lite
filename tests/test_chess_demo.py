from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import chess_demo  # noqa: E402
from systemone_lite import SystemOneClient  # noqa: E402
from systemone_lite.stub import StubEngine  # noqa: E402


def test_chess_game_initialization():
    game = chess_demo.ProperChessGame()
    assert game.plies == 0
    assert not game.game_over
    w_mat, b_mat = game.get_material_count()
    assert w_mat == 39
    assert b_mat == 39


def test_analyze_legal_moves_opening():
    game = chess_demo.ProperChessGame()
    candidates = chess_demo.analyze_legal_moves(game.board)
    assert len(candidates) == 20  # 20 legal moves in starting chess position

    # All legal opening moves should be valid moves
    all_san = {c.san for c in candidates}
    assert "e4" in all_san
    assert "d4" in all_san
    assert "Nf3" in all_san


def test_chess_demo_stub_play():
    client = SystemOneClient(engine=StubEngine())
    res = chess_demo.play_game(
        client=client,
        max_plies=4,
        delay=0.0,
        animate=False,
    )
    assert res["plies"] == 4
    assert len(res["history"]) == 4
    assert isinstance(res["history"][0], str)

