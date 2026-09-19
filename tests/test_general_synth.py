"""Tests for general synthetic gyms."""

from __future__ import annotations

import random

from systemone_lite.synth import (
    generate_allocator_episode,
    generate_debate_episode,
    generate_ticket_episode,
)
from systemone_lite.synth.common import maybe_subset_options, perturb_state


def _assert_samples(samples) -> None:
    assert len(samples) >= 2
    for s in samples:
        assert s.label_alias in s.criteria
        assert s.label_key
        assert s.instructions
        assert isinstance(s.state, dict)


def test_ticket_dungeon_labels_consistent() -> None:
    rng = random.Random(0)
    samples = generate_ticket_episode(rng)
    _assert_samples(samples)
    tasks = {s.task for s in samples}
    assert "ticket.route" in tasks
    assert "ticket.needs_human" in tasks
    assert "ticket.urgency" in tasks


def test_hard_variants() -> None:
    for seed in range(20):
        rng = random.Random(seed)
        _assert_samples(generate_ticket_episode(rng, hard=True))
        _assert_samples(generate_allocator_episode(rng, hard=True))
        _assert_samples(generate_debate_episode(rng, hard=True))


def test_allocator_and_debate() -> None:
    rng = random.Random(1)
    _assert_samples(generate_allocator_episode(rng))
    _assert_samples(generate_debate_episode(rng))


def test_billing_ticket_routes_to_billing() -> None:
    found = False
    for seed in range(300):
        samples = generate_ticket_episode(random.Random(seed))
        route = next(s for s in samples if s.task == "ticket.route")
        if route.meta.get("domain_seed") == "billing":
            assert route.label_key == "billing"
            found = True
            break
    assert found


def test_option_subset_keeps_label() -> None:
    rng = random.Random(0)
    opts = {"a": "A", "b": "B", "c": "C", "d": "D"}
    for _ in range(50):
        sub = maybe_subset_options(rng, opts, "b", hard=True)
        assert "b" in sub
        assert 2 <= len(sub) <= 4


def test_perturb_state_hard_can_wrap() -> None:
    rng = random.Random(2)
    wrapped = False
    for _ in range(40):
        out = perturb_state(rng, {"x": 1, "y": 2}, hard=True)
        if "payload" in out or "application_state" in out:
            wrapped = True
            break
    assert wrapped
