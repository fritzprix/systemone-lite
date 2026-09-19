from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from systemone_lite.confidence import distribution_confidence
from systemone_lite.schema import SystemOneRequest, SystemOneResponse
from systemone_lite.stub import StubEngine

FIXTURES = Path(__file__).parent / "fixtures"


def test_official_request_parses() -> None:
    raw = json.loads((FIXTURES / "official_example_request.json").read_text())
    req = SystemOneRequest.model_validate(raw)
    assert set(req.questions) == {"is_urgent", "department", "frustration"}
    assert req.questions["is_urgent"].type == "noul"
    assert req.questions["department"].type == "choice"
    assert req.questions["frustration"].type == "score"


def test_questions_array_rejected() -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(
            {
                "state": "x",
                "questions": [
                    {"type": "noul", "instructions": "yes?"},
                ],
            }
        )


def test_choice_requires_criteria() -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(
            {
                "state": "x",
                "questions": {
                    "q": {"type": "choice", "instructions": "pick"},
                },
            }
        )


def test_score_requires_two_levels() -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(
            {
                "state": "x",
                "questions": {
                    "q": {
                        "type": "score",
                        "instructions": "rate",
                        "criteria": ["only-one"],
                    },
                },
            }
        )


def test_stub_response_shape_matches_jev() -> None:
    raw = json.loads((FIXTURES / "official_example_request.json").read_text())
    req = SystemOneRequest.model_validate(raw)
    resp = StubEngine().decide(req)

    # Round-trip through response model (extra fields forbidden).
    validated = SystemOneResponse.model_validate(resp.model_dump())
    payload = validated.model_dump(mode="json")

    assert set(payload.keys()) == {"model", "answers", "usage"}
    assert set(payload["answers"].keys()) == set(req.questions.keys())
    assert set(payload["usage"].keys()) == {"input_tokens", "output_tokens"}
    assert payload["usage"]["output_tokens"] == 0

    noul = payload["answers"]["is_urgent"]
    assert set(noul.keys()) == {"type", "noul"}
    assert "confidence" not in noul
    assert 0.0 <= noul["noul"] <= 1.0

    choice = payload["answers"]["department"]
    assert set(choice.keys()) == {"type", "choice", "probabilities", "confidence"}
    assert choice["choice"] in choice["probabilities"]
    assert abs(sum(choice["probabilities"].values()) - 1.0) < 1e-6

    score = payload["answers"]["frustration"]
    assert set(score.keys()) == {
        "type",
        "score",
        "legend",
        "probabilities",
        "confidence",
    }
    assert set(score["legend"].keys()) == {"0", "1", "2"}
    assert set(score["probabilities"].keys()) == {"0", "1", "2"}
    assert abs(sum(score["probabilities"].values()) - 1.0) < 1e-6


def test_confidence_formula() -> None:
    # n=3, p_max=0.8 → (0.8 - 1/3) / (2/3) ≈ 0.7
    conf = distribution_confidence({"a": 0.8, "b": 0.1, "c": 0.1})
    assert conf == pytest.approx(0.7, abs=1e-6)
