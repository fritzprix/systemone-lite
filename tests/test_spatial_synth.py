"""Unit tests for spatial 2D game synthetic distillation generators."""

from __future__ import annotations

from systemone_lite.synth.connect4 import generate_connect4_samples
from systemone_lite.synth.game2048 import generate_2048_samples
from systemone_lite.synth.gridworld import generate_gridworld_samples
from systemone_lite.synth.sokoban import generate_sokoban_samples


def test_sokoban_generator() -> None:
    samples = generate_sokoban_samples(10, seed=123)
    assert len(samples) == 10
    for s in samples:
        assert s.meta["gym"] == "sokoban"
        assert "grid_map" in s.state
        assert s.label_alias in s.criteria
        assert "#" in s.state["grid_map"]


def test_2048_generator() -> None:
    samples = generate_2048_samples(10, seed=456)
    assert len(samples) == 10
    for s in samples:
        assert s.meta["gym"] == "game2048"
        assert "grid_map" in s.state
        assert s.label_alias in s.criteria


def test_gridworld_generator() -> None:
    samples = generate_gridworld_samples(10, seed=789)
    assert len(samples) == 10
    for s in samples:
        assert s.meta["gym"] == "gridworld"
        assert "grid_map" in s.state
        assert s.label_alias in s.criteria
        assert "@" in s.state["grid_map"]


def test_connect4_generator() -> None:
    samples = generate_connect4_samples(10, seed=999)
    assert len(samples) == 10
    for s in samples:
        assert s.meta["gym"] == "connect4"
        assert "grid_map" in s.state
        assert s.label_alias in s.criteria
