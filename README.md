# systemone-lite

A **local System One–compatible decision API** on a **0.5B** language model.

You send application `state` plus typed questions (`yes/no`, multiple choice, or
ordered score). The server returns **one discrete answer per question** by scoring
option tokens only — not by generating JSON prose.

- **API:** `POST /v1/systemone` (same request/response shape as the public
  [System One contract](https://docs.typesafe.ai/api.md))
- **Model:** [`dwidlee/systemone-lite-spatial-v2-s1`](https://huggingface.co/dwidlee/systemone-lite-spatial-v2-s1)
  (fine-tuned from `Qwen/Qwen2.5-0.5B-Instruct`)
- **Hardware:** consumer GPU friendly (latency numbers below: RTX 3060)

> Independent open-source project. **Not** affiliated with TypeSafe AI / Jev,
> and **not** a drop-in cloud replacement.

---

## Install

```bash
git clone https://github.com/fritzprix/systemone-lite.git
cd systemone-lite
pip install -e ".[dev]"
```

Python 3.11+ · GPU recommended.

---

## Use it

### Server

```bash
systemone-lite --model dwidlee/systemone-lite-spatial-v2-s1 --port 8000

curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @tests/fixtures/official_example_request.json
```

### Python

```python
from systemone_lite import SystemOneClient, choice, noul, score

client = SystemOneClient(model="dwidlee/systemone-lite-spatial-v2-s1")

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

print(response.answers["route"].choice)       # e.g. "billing"
print(response.answers["needs_review"].noul)  # True / False
print(response.answers["urgency"].score)      # e.g. "high"
```

OpenAPI sketch: [`openapi/systemone.yaml`](openapi/systemone.yaml).

---

## How good is it?

Published weights, **local** measurements (not an official leaderboard submission).

| What | Number |
|---|---|
| JevBench-style public set (231 tasks) | **49.8%** accuracy · ECE **0.307** · p50 **12.6 ms** |
| Short 1-question payloads | typically **~10–30 ms** |
| Large state / many questions | typically **~100–160 ms** |
| vs greedy AR JSON (same weights) | roughly **10–50×** faster option scoring |

Example rollouts (illustration only): [`benchmarks/demos/spatial_v2_s1/`](benchmarks/demos/spatial_v2_s1/).

Raw reports: [`benchmarks/jevbench_spatial_v2_s1.json`](benchmarks/jevbench_spatial_v2_s1.json),
[`benchmarks/latency_vs_ar.json`](benchmarks/latency_vs_ar.json).

---

## Limits (read this)

- **Small model, single-token answers.** Good for short typed decisions (routing,
  yes/no, pick-one). Bad at long multi-step planning; puzzle/game rollouts still
  fail often even when snapshot accuracy looks fine.
- **Calibration is mediocre.** Accuracies above are usable for demos; do not treat
  returned probabilities as well-calibrated confidence.
- **Different from production Jev.** Own weights, own scoring path, own
  `confidence` formula `(p_max - 1/n)/(1 - 1/n)`.
- **Option keys in training are letter aliases.** Multi-token option ids are not
  first-class.
- **Eval:** use the Hub dataset **`test`** split for held-out numbers — never score
  on `train`.

Training data:
[`dwidlee/systemone-lite-phase2`](https://huggingface.co/datasets/dwidlee/systemone-lite-phase2)
(240 800 train / 4 700 test).

---

## Optional: demos & retrain

```bash
pytest
python scripts/demo_systemone.py
```

Retrain / rebuild docs and research notes: [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## License

MIT

## Acknowledgments

System One / Jev concepts and public API docs belong to their respective owners
(TypeSafe AI). This repo is an unofficial local approximation for experimentation.
