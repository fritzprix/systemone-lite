"""Guards for schedule-horizon / LR continuity on resume (GitHub #8)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from chess_finetune import (  # noqa: E402
    assert_lr_continuity,
    assert_schedule_horizon,
    configs_compatible,
)


def test_configs_compatible_requires_exact_max_steps() -> None:
    base = {
        "data": "/tmp/train.jsonl",
        "model_id": "checkpoints/x",
        "epochs": 1,
        "batch_size": 4,
        "lr": 2e-5,
        "max_length": 768,
        "max_steps": 20000,
        "seed": 7,
        "tasks": ["all"],
        "stratified": True,
    }
    assert configs_compatible(base, dict(base))
    raised = dict(base)
    raised["max_steps"] = 60200
    assert not configs_compatible(base, raised)


def test_assert_schedule_horizon_rejects_change() -> None:
    assert_schedule_horizon(saved_total_steps=20000, total_steps=20000)
    with pytest.raises(SystemExit, match="horizon"):
        assert_schedule_horizon(saved_total_steps=20000, total_steps=60200)


def test_assert_lr_continuity_rejects_jump() -> None:
    assert_lr_continuity(saved_lr=7.78e-8, current_lr=7.78e-8)
    with pytest.raises(SystemExit, match="LR discontinuity"):
        assert_lr_continuity(saved_lr=7.78e-8, current_lr=1.48e-5)


def test_assert_lr_continuity_skips_legacy_missing() -> None:
    assert_lr_continuity(saved_lr=None, current_lr=1e-5)
