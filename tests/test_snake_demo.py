from __future__ import annotations

import sys
from pathlib import Path

# Add scripts directory to path to import snake_demo
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import snake_demo  # noqa: E402
from systemone_lite import SystemOneClient  # noqa: E402
from systemone_lite.stub import StubEngine  # noqa: E402


def test_snake_game_initialization():
    game = snake_demo.SnakeGame(width=8, height=8, seed=42)
    assert len(game.snake) == 3
    assert game.head == (4, 4)
    assert game.score == 0
    assert not game.game_over


def test_snake_game_step_wall_collision():
    game = snake_demo.SnakeGame(width=4, height=4, seed=42)
    game.snake = [(0, 0), (0, 1), (0, 2)]
    # Moving UP from row 0 should hit wall
    moved = game.step("UP")
    assert not moved
    assert game.game_over
    assert "Hit wall" in game.death_reason


def test_snake_game_step_self_collision():
    game = snake_demo.SnakeGame(width=6, height=6, seed=42)
    # Loop snake with length 5: head at (2, 2), body covers (3, 2), tail is at (3, 1)
    game.snake = [(2, 2), (2, 3), (3, 3), (3, 2), (3, 1)]
    # Moving DOWN into (3, 2) is a self collision with body segment
    moved = game.step("DOWN")
    assert not moved
    assert game.game_over
    assert "own body" in game.death_reason


def test_snake_game_eat_food():
    game = snake_demo.SnakeGame(width=8, height=8, seed=42)
    game.snake = [(3, 3), (3, 2), (3, 1)]
    game.food = (3, 4)
    moved = game.step("RIGHT")
    assert moved
    assert game.score == 1
    assert len(game.snake) == 4
    assert game.head == (3, 4)


def test_snake_payload_and_stub_episode():
    game = snake_demo.SnakeGame(width=8, height=8, seed=123)
    state, questions, alias_to_dir = game.build_systemone_payload()

    assert "direction" in questions
    assert "danger_level" in questions
    assert "food_reachable" in questions
    assert len(alias_to_dir) == 4

    client = SystemOneClient(engine=StubEngine())
    result = snake_demo.run_episode(
        client=client,
        game=game,
        delay=0.0,
        max_steps=5,
        animate=False,
    )
    assert "score" in result
    assert "steps" in result
    assert result["steps"] <= 5
    assert result["total_decisions"] > 0
