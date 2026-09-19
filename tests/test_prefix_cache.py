"""Prefix KV cache correctness vs naive full-prompt batching."""

from __future__ import annotations

import pytest

from systemone_lite.infer import SystemOneEngine, get_engine, load_model, reset_engine
from systemone_lite.schema import SystemOneRequest


@pytest.fixture(scope="module")
def loaded():
    reset_engine()
    try:
        return load_model("systemone-lite-latest")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"model unavailable: {exc}")


def test_prefix_cache_matches_naive_choice(loaded) -> None:
    request = SystemOneRequest.model_validate(
        {
            "model": "systemone-lite-latest",
            "state": "Card charged twice; customer wants a refund.",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {
                        "billing": "charges / refunds",
                        "technical": "bugs",
                        "other": None,
                    },
                },
                "urgent": {
                    "type": "noul",
                    "instructions": "Is this urgent?",
                },
                "sev": {
                    "type": "score",
                    "instructions": "Severity",
                    "criteria": ["low", "medium", "high"],
                },
            },
        }
    )

    cached = SystemOneEngine(loaded, use_prefix_cache=True).decide(request)
    naive = SystemOneEngine(loaded, use_prefix_cache=False).decide(request)

    assert set(cached.answers) == set(naive.answers)
    assert cached.answers["route"].choice == naive.answers["route"].choice
    assert abs(cached.answers["urgent"].noul - naive.answers["urgent"].noul) < 0.05
    # Score can drift slightly with fp16 + cache path; keep a loose bound.
    assert abs(cached.answers["sev"].score - naive.answers["sev"].score) < 0.15

    # Prefix cache should count state once.
    assert cached.usage.input_tokens < naive.usage.input_tokens


def test_get_engine_defaults_to_prefix_cache() -> None:
    reset_engine()
    engine = get_engine("systemone-lite-latest")
    assert engine.use_prefix_cache is True
