"""Thin HTTP / in-process client for System One."""

from __future__ import annotations

from typing import Any

import httpx

from systemone_lite.infer import SystemOneEngine, get_engine
from systemone_lite.schema import Question, SystemOneRequest, SystemOneResponse


def noul(
    instructions: str,
    criteria: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria is not None:
        payload["criteria"] = criteria
    return payload


def choice(instructions: str, criteria: dict[str, str | None]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, criteria: list[str]) -> dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": criteria}


class SystemOneClient:
    """Call a remote server, or run in-process when base_url is None."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "systemone-lite-latest",
        engine: SystemOneEngine | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.model = model
        self._engine = engine

    def system_one(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, Question | dict[str, Any]],
        model: str | None = None,
    ) -> SystemOneResponse:
        request = SystemOneRequest.model_validate(
            {
                "model": model or self.model,
                "state": state,
                "questions": questions,
            }
        )
        if self.base_url is None:
            engine = self._engine or get_engine(request.model)
            return engine.decide(request)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/v1/systemone",
                json=request.model_dump(mode="json"),
                headers=headers,
            )
            response.raise_for_status()
            return SystemOneResponse.model_validate(response.json())
