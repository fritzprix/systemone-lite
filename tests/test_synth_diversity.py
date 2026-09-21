"""Unit tests for CA and word-game synth (#7)."""

from __future__ import annotations

from systemone_lite.synth.cellular_automata import (
    CAGrid,
    RULE_PRESETS,
    generate_ca_samples,
)
from systemone_lite.synth.word_games import generate_word_game_samples


def test_conway_blinker_period_two():
    # Vertical blinker in 5x5
    cells = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    birth, survive = RULE_PRESETS["B3/S23"]
    g = CAGrid(cells, birth, survive, "B3/S23")
    g2 = g.after(2)
    assert g2.cells == cells


def test_generate_ca_samples_schema():
    samples = generate_ca_samples(60, seed=0)
    assert len(samples) == 60
    tasks = {s.task for s in samples}
    assert "ca.cell_alive" in tasks
    assert "ca.pop_parity" in tasks
    assert "ca.pop_change" in tasks
    for s in samples:
        assert s.label_alias in s.criteria
        assert s.meta.get("gym") == "cellular_automata"
        assert "legend" in s.state
        assert "n_steps" in s.state
        dump = s.to_json()
        assert "RECOMMENDED" not in str(dump)


def test_generate_word_game_samples_schema():
    samples = generate_word_game_samples(50, seed=1)
    assert len(samples) == 50
    kinds = {s.meta.get("kind") for s in samples}
    assert kinds >= {"anagram", "definition", "plural"}
    for s in samples:
        assert s.meta.get("gym") == "word_games"
        assert s.label_alias in s.criteria
        assert s.label_key is not None
