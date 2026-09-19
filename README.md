# systemone-lite

> A **toy / research project** that approximates TypeSafe-style **System One** decisions
> with a small open model you can run **locally**.
>
> Not affiliated with [TypeSafe AI](https://typesafe.ai) or Jev. Not a drop-in production replacement.

`systemone-lite` exposes a **Jev-shaped** HTTP API (`POST /v1/systemone`): you send application `state` plus typed questions (`noul` / `choice` / `score`), and get back constrained answers with probabilities.

Under the hood it is **not** Jev’s proprietary parallel sampler or RLCD stack. It is a lightweight causal LM (`Qwen/Qwen2.5-0.5B-Instruct` by default) that:

1. Builds a shared state prefix + per-question templates  
2. Scores option tokens with a single next-token step (batched)  
3. Reuses a **prefix KV cache** so long state is encoded once across questions  

Think of it as an educational / hackable **local System One sandbox**, useful for API prototyping, latency experiments, and learning how typed decision APIs feel in application code.

## Why this exists

- **Local-first:** run on your GPU/CPU; no TypeSafe account required for development  
- **Wire-compatible shape:** same request/response ideas as the [public System One API](https://docs.typesafe.ai/api.md) so client code can be sketched offline  
- **Tiny backbone:** start with ~0.5B params; swap models later if you want  
- **Honest scope:** accuracy and calibration will not match a frontier System One product out of the box—especially before fine-tuning  

## What it is / isn’t

| Is | Isn’t |
|---|---|
| Toy approximation of System One **API contracts** | A clone of Jev’s model, sampler, or training |
| Local FastAPI server + Python client | Hosted SaaS or calibrated enterprise model |
| Batched AR + option logits + prefix KV cache | BERT/MLM multi-mask “parallel MLM” |
| MIT open source | Affiliated with or endorsed by TypeSafe |

## Requirements

- Python 3.11+  
- `torch`, `transformers`, `fastapi` (see `pyproject.toml`)  
- GPU recommended (smoke-tested on RTX 3060); CPU works but slower  

## Install

```bash
git clone https://github.com/fritzprix/systemone-lite.git
cd systemone-lite
pip install -e ".[dev]"
```

## Quick start (local)

### 1) Unit tests (no model download)

```bash
pytest
```

### 2) Live demo (downloads ~0.5B weights on first run)

```bash
python scripts/demo_systemone.py
```

### 3) Local API server

```bash
systemone-lite --host 127.0.0.1 --port 8000
# or: python -m systemone_lite.api
```

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @tests/fixtures/official_example_request.json
```

Point any System One–shaped client at `http://127.0.0.1:8000` instead of TypeSafe’s host when experimenting locally.

## Python client

```python
from systemone_lite import SystemOneClient, choice, noul, score

# In-process (no HTTP), or: SystemOneClient(base_url="http://127.0.0.1:8000")
client = SystemOneClient()

response = client.system_one(
    state="My card was charged twice.",
    questions={
        "needs_review": noul("Does this need a human agent?"),
        "route": choice(
            "Route to a team",
            {"billing": "charges", "technical": "bugs", "other": None},
        ),
        "urgency": score("Urgency", ["low", "medium", "high"]),
    },
)

print(response.answers["route"].choice)
print(response.answers["needs_review"].noul)
print(response.answers["urgency"].score)
```

## API compatibility (approximate)

Compatible with the public System One **JSON shape**:

- `POST /v1/systemone`
- `state`: string | object | array  
- `questions`: **map** of `noul` | `choice` | `score`  
- `answers` + `usage.input_tokens` / `usage.output_tokens`  

Intentionally different:

| Topic | Notes |
|---|---|
| Model ids | `systemone-lite-latest` → Qwen 0.5B (not `jev-*`) |
| Quality / calibration | Toy baseline; fine-tune if you care about accuracy |
| `confidence` | Derived as `(p_max - 1/n) / (1 - 1/n)`; may differ from TypeSafe |
| Auth / pricing | Local; no TypeSafe billing |

OpenAPI sketch: [`openapi/systemone.yaml`](openapi/systemone.yaml)  
Design notes: [`docs/PROPOSAL.md`](docs/PROPOSAL.md)

## Latency (indicative)

In-process on an **RTX 3060**, `Qwen2.5-0.5B-Instruct`, with prefix KV cache (warmup excluded):

| Case | p50 latency |
|---|---:|
| Short state · 3 questions | ~25 ms |
| Long state (~6k chars) · 13 questions | ~145 ms |

TypeSafe publicly cites Jev E2E latencies roughly in the **70–500 ms** band (their cloud + network). Numbers are **not** a head-to-head bake-off: different hardware, no shared eval set, and this project prioritizes local hackability over product parity.

Reproduce:

```bash
python scripts/bench_latency.py --warmup 5 --runs 20
```

## Project layout

```text
src/systemone_lite/   # schema, prompt, infer (prefix cache), API, client
tests/                # shape / API / prefix-cache checks
scripts/              # demo + latency bench
openapi/              # System One–shaped OpenAPI
docs/PROPOSAL.md      # design decisions
```

## Status

Early toy project. Expect rough edges: multi-token option labels, weak zero-shot accuracy, and limited score semantics until you add data/fine-tuning.

Contributions and experiments welcome—especially calibration, better option tokenization, and cleaner serving.

## License

MIT

## Acknowledgments / disclaimer

System One / Jev concepts and public API docs belong to their respective owners (TypeSafe AI). This repository is an independent, unofficial approximation for local experimentation only.
