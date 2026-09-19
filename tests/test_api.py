from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from systemone_lite import api
from systemone_lite.stub import StubEngine

FIXTURES = Path(__file__).parent / "fixtures"


def test_systemone_endpoint_with_stub(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_engine", lambda _model: StubEngine())
    client = TestClient(api.app)

    raw = json.loads((FIXTURES / "official_example_request.json").read_text())
    response = client.post("/v1/systemone", json=raw)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "stub-systemone-lite"
    assert "is_urgent" in body["answers"]
    assert body["answers"]["is_urgent"]["type"] == "noul"
    assert "confidence" not in body["answers"]["is_urgent"]


def test_systemone_rejects_questions_array(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_engine", lambda _model: StubEngine())
    client = TestClient(api.app)
    response = client.post(
        "/v1/systemone",
        json={
            "state": "x",
            "questions": [{"type": "noul", "instructions": "yes?"}],
        },
    )
    assert response.status_code == 422


def test_health() -> None:
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok"}
