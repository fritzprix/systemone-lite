"""FastAPI server exposing Jev-compatible POST /v1/systemone."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from systemone_lite.infer import DEFAULT_MODEL_ID, get_engine, set_default_model
from systemone_lite.schema import SystemOneRequest, SystemOneResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="systemone-lite",
    description="Jev-compatible typed decision API (lightweight causal LM).",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/systemone", response_model=SystemOneResponse)
def systemone(body: dict[str, Any]) -> SystemOneResponse:
    try:
        request = SystemOneRequest.model_validate(body)
    except ValidationError as exc:
        detail = json.loads(exc.json())
        raise HTTPException(status_code=422, detail=detail) from exc

    try:
        engine = get_engine(request.model)
        return engine.decide(request)
    except Exception as exc:  # noqa: BLE001 — surface as 500 with message
        logger.exception("systemone failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run systemone-lite API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    set_default_model(args.model)
    # Warm the default engine at startup when not reloading.
    if not args.reload:
        get_engine(args.model)

    import uvicorn

    uvicorn.run(
        "systemone_lite.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
