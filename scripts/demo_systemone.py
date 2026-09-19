#!/usr/bin/env python3
"""Run the official-shaped System One example against the local engine."""

from __future__ import annotations

import json
from pathlib import Path

from systemone_lite import SystemOneClient

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "official_example_request.json"


def main() -> None:
    raw = json.loads(FIXTURE.read_text())
    client = SystemOneClient(model=raw.get("model", "systemone-lite-latest"))
    response = client.system_one(state=raw["state"], questions=raw["questions"])
    print(json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
