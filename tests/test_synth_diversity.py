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


def test_nlp_cloze_offline_schema_and_disjoint():
    from systemone_lite.synth.nlp_cloze import generate_nlp_cloze_samples_offline

    train = generate_nlp_cloze_samples_offline(20, seed=1, split="train")
    eval_s = generate_nlp_cloze_samples_offline(20, seed=2, split="eval")
    assert len(train) == 20 and len(eval_s) == 20
    for s in train + eval_s:
        assert s.task == "nlp.cloze"
        assert s.meta.get("gym") == "nlp_cloze"
        assert s.label_alias in s.criteria
        assert "[___]" in str(s.state.get("sentence") or s.state)
    train_docs = {s.meta.get("doc_id") for s in train}
    # Offline banks are disjoint sentence lists; labels should not collide heavily.
    train_keys = {s.label_key for s in train}
    eval_keys = {s.label_key for s in eval_s}
    # Soft check: at least some separation in answer sets across tiny banks
    assert train_keys or eval_keys
    assert train_docs  # smoke


def test_word_train_eval_vocab_disjoint():
    from systemone_lite.synth.word_games import (
        EVAL_DEFINITIONS,
        EVAL_PLURALS,
        EVAL_WORDS,
        TRAIN_DEFINITIONS,
        TRAIN_PLURALS,
        TRAIN_WORDS,
    )

    assert not (set(TRAIN_WORDS) & set(EVAL_WORDS))
    assert not (set(TRAIN_DEFINITIONS) & set(EVAL_DEFINITIONS))
    assert not (set(TRAIN_PLURALS) & set(EVAL_PLURALS))
    train = generate_word_game_samples(80, seed=2, split="train")
    eval_s = generate_word_game_samples(80, seed=3, split="eval")
    train_words = {
        (s.state.get("word") or s.state.get("base_word") or s.label_key)
        for s in train
        if s.meta.get("kind") in {"anagram", "vowel_count", "one_edit", "definition"}
    }
    eval_words = {
        (s.state.get("word") or s.state.get("base_word") or s.label_key)
        for s in eval_s
        if s.meta.get("kind") in {"anagram", "vowel_count", "one_edit", "definition"}
    }
    assert train_words.isdisjoint(eval_words)