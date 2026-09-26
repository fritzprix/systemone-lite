"""Unit tests for spatial 2D game synthetic distillation generators."""

from __future__ import annotations

import re
from collections import Counter

from systemone_lite.synth.connect4 import generate_connect4_samples
from systemone_lite.synth.game2048 import generate_2048_samples
from systemone_lite.synth.gridworld import generate_gridworld_samples
from systemone_lite.synth.sokoban import (
    generate_sokoban_eval_balanced,
    generate_sokoban_samples,
)


def _player_glyph(sample) -> str:
    remap = (sample.meta or {}).get("symbol_remap")
    if isinstance(remap, dict) and remap.get("player"):
        return str(remap["player"])
    legend = str(sample.state.get("legend") or "")
    m = re.search(r"'([^']+)':\s*Player", legend)
    return m.group(1) if m else "@"


def test_sokoban_generator() -> None:
    samples = generate_sokoban_samples(10, seed=123)
    assert len(samples) == 10
    for s in samples:
        assert s.meta["gym"] == "sokoban"
        assert "grid_map" in s.state
        assert "legend" in s.state
        assert s.label_alias in s.criteria
        assert _player_glyph(s) in s.state["grid_map"] or "$" in s.state["grid_map"]


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
        assert "legend" in s.state
        assert s.label_alias in s.criteria
        assert _player_glyph(s) in s.state["grid_map"]


def test_connect4_generator() -> None:
    samples = generate_connect4_samples(10, seed=999)
    assert len(samples) == 10
    for s in samples:
        assert s.meta["gym"] == "connect4"
        assert "grid_map" in s.state
        assert s.label_alias in s.criteria


def test_connect4_action_v2_caps_drop_options() -> None:
    samples = generate_connect4_samples(80, seed=999)
    drops = [s for s in samples if s.task == "connect4.drop"]
    assert drops
    assert all(len(s.criteria) <= 3 for s in drops)
    assert any(s.task == "connect4.win_now_alert" for s in samples)
    assert any(s.meta.get("task_schema") == "action_v2" for s in samples)


def test_sokoban_eval_balanced() -> None:
    rows = generate_sokoban_eval_balanced(100, seed=7, n_alerts=40)
    assert len(rows) == 100
    tasks = Counter(r["task"] for r in rows)
    assert tasks["sokoban.deadlock_alert"] == 40
    yes = sum(
        1
        for r in rows
        if r["task"] == "sokoban.deadlock_alert" and r["label_key"] == "yes"
    )
    no = sum(
        1
        for r in rows
        if r["task"] == "sokoban.deadlock_alert" and r["label_key"] == "no"
    )
    assert yes == 20 and no == 20
    assert all(r["meta"].get("eval_balanced") for r in rows)


def test_spatial_action_v2_legal_only() -> None:
    for gen, gym in [
        (generate_2048_samples, "game2048"),
        (generate_gridworld_samples, "gridworld"),
        (generate_sokoban_samples, "sokoban"),
    ]:
        samples = gen(40, seed=42)
        actions = [
            s
            for s in samples
            if s.task.endswith((".slide", ".move", ".direction"))
        ]
        assert actions, gym
        for s in actions:
            assert len(s.criteria) <= 4
            assert s.meta.get("task_schema") == "action_v2"
