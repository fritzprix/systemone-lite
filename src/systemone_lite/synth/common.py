"""Shared helpers for general System One synthetic distillation."""

from __future__ import annotations

import random
from typing import Any

from systemone_lite.chess_data import DistillSample, alias_criteria


def shuffle_mapping(rng: random.Random, mapping: dict[str, Any]) -> dict[str, Any]:
    items = list(mapping.items())
    rng.shuffle(items)
    return dict(items)


def perturb_state(rng: random.Random, state: dict[str, Any], *, hard: bool) -> dict[str, Any]:
    """Shuffle key order; optionally wrap / rename top-level layout for hard split."""
    base = shuffle_mapping(rng, state)
    if not hard:
        return base
    mode = rng.choice(["wrap", "rename", "flat", "plain"])
    if mode == "wrap":
        return {"payload": base, "format": "wrapped"}
    if mode == "rename":
        return {"application_state": base, "version": 1}
    if mode == "flat" and len(base) == 1:
        only = next(iter(base.values()))
        if isinstance(only, dict):
            return shuffle_mapping(rng, only)
    return base


def maybe_subset_options(
    rng: random.Random,
    options: dict[str, str],
    label_key: str,
    *,
    hard: bool,
) -> dict[str, str]:
    """Keep the label; randomly drop distractors for format diversity."""
    if label_key not in options:
        raise KeyError(label_key)
    if len(options) <= 2:
        return dict(options)
    # Soft: sometimes drop one distractor. Hard: drop more aggressively.
    drop_p = 0.35 if hard else 0.15
    if rng.random() > drop_p:
        return dict(options)
    keep = {label_key: options[label_key]}
    others = [k for k in options if k != label_key]
    rng.shuffle(others)
    min_keep = 1 if hard else max(1, len(others) - 1)
    n_keep = rng.randint(min_keep, len(others))
    for k in others[:n_keep]:
        keep[k] = options[k]
    return keep


def cap_options(
    options: dict[str, str],
    label_key: str,
    *,
    max_n: int,
    rng: random.Random,
) -> dict[str, str]:
    """Keep ``label_key`` plus random distractors, hard-capped at ``max_n``."""
    if label_key not in options:
        raise KeyError(label_key)
    if max_n < 1:
        raise ValueError("max_n must be >= 1")
    if len(options) <= max_n:
        return dict(options)
    others = [k for k in options if k != label_key]
    rng.shuffle(others)
    keep_keys = [label_key] + others[: max_n - 1]
    rng.shuffle(keep_keys)
    return {k: options[k] for k in keep_keys}


TASK_SCHEMA_ACTION_V2 = "action_v2"


def choice_sample(
    *,
    task: str,
    state: dict[str, Any],
    instructions: str,
    options: dict[str, str],
    label_key: str,
    meta: dict[str, Any],
    rng: random.Random,
    hard: bool = False,
) -> DistillSample:
    """Build a single-token-alias choice sample with option/state diversity."""
    state = perturb_state(rng, state, hard=hard)
    options = maybe_subset_options(rng, options, label_key, hard=hard)
    items = list(options.items())
    rng.shuffle(items)
    ordered = dict(items)
    if label_key not in ordered:
        raise KeyError(f"label_key {label_key!r} missing from options")

    criteria, alias_to_key = alias_criteria(ordered)
    key_to_alias = {v: k for k, v in alias_to_key.items()}
    return DistillSample(
        task=task,
        state=state,
        instructions=instructions,
        criteria=criteria,
        label_alias=key_to_alias[label_key],
        label_key=label_key,
        meta={
            **meta,
            "schema": "choice",
            "gym": meta.get("gym", "general"),
            "hard": hard,
            "n_options": len(ordered),
            "task_schema": meta.get("task_schema", "choice"),
        },
    )


def paraphrase(rng: random.Random, templates: list[str], **kwargs: str) -> str:
    return templates[rng.randrange(len(templates))].format(**kwargs)
