"""Shared helpers for general System One synthetic distillation."""

from __future__ import annotations

import random
from typing import Any

from systemone_lite.chess_data import DistillSample, alias_criteria


def choice_sample(
    *,
    task: str,
    state: dict[str, Any],
    instructions: str,
    options: dict[str, str],
    label_key: str,
    meta: dict[str, Any],
    rng: random.Random,
) -> DistillSample:
    """Build a single-token-alias choice sample; optionally drop/add distractors."""
    # Keep label, shuffle option order for format diversity.
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
        meta={**meta, "schema": "choice", "gym": meta.get("gym", "general")},
    )


def paraphrase(rng: random.Random, templates: list[str], **kwargs: str) -> str:
    return templates[rng.randrange(len(templates))].format(**kwargs)
