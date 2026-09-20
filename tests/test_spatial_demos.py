"""Unit tests for spatial 2D game demos running with StubEngine."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from systemone_lite import SystemOneClient
from systemone_lite.stub import StubEngine
from sokoban_demo import play_sokoban_demo
from game2048_demo import play_2048_demo
from gridworld_demo import play_gridworld_demo
from connect4_demo import play_connect4_demo
from chess_multistep_demo import play_multistep_game


def test_sokoban_demo_stub() -> None:
    client = SystemOneClient(engine=StubEngine())
    res = play_sokoban_demo(client, max_steps=5, delay=0.0, animate=False)
    assert res["steps"] > 0


def test_2048_demo_stub() -> None:
    client = SystemOneClient(engine=StubEngine())
    res = play_2048_demo(client, max_steps=6, delay=0.0, animate=False)
    assert res["steps"] > 0
    assert res["max_tile"] >= 2


def test_gridworld_demo_stub() -> None:
    client = SystemOneClient(engine=StubEngine())
    res = play_gridworld_demo(client, max_steps=5, delay=0.0, animate=False)
    assert res["steps"] > 0


def test_connect4_demo_stub() -> None:
    client = SystemOneClient(engine=StubEngine())
    res = play_connect4_demo(client, max_turns=6, delay=0.0, animate=False)
    assert res["turns"] > 0


def test_chess_multistep_demo_stub() -> None:
    client = SystemOneClient(engine=StubEngine())
    res = play_multistep_game(client, max_plies=4, delay=0.0, step_delay=0.0, animate=False)
    assert res["plies"] == 4
    assert len(res["history"]) == 4
