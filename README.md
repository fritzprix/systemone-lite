# systemone-lite

A **local System One–compatible decision API** on a **0.5B** language model.

You send application `state` plus typed questions (`yes/no`, multiple choice, or
ordered score). The server returns **one discrete answer per question** by scoring
option tokens only — not by generating JSON prose.

- **API:** `POST /v1/systemone` (same request/response shape as the public
  [System One contract](https://docs.typesafe.ai/api.md))
- **Model:** [`dwidlee/systemone-lite-0.5b`](https://huggingface.co/dwidlee/systemone-lite-0.5b)
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
systemone-lite --model dwidlee/systemone-lite-0.5b --port 8000

curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @tests/fixtures/official_example_request.json
```

### Python

```python
from systemone_lite import SystemOneClient, choice, noul, score

client = SystemOneClient(model="dwidlee/systemone-lite-0.5b")

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

Published weights (`dwidlee/systemone-lite-0.5b`, revision **action-v2-qwen**),
**local** measurements (not an official leaderboard submission).
n=800 SE ≈ ±1.8%p; n=231 SE ≈ ±3%p — small deltas are noise.

| What | Number |
|---|---|
| Phase2 held-out (`test`, n=4700) | **61.6%** |
| First-800 protocol (n=800) | **63.9%** |
| JevBench public (231 tasks, T=1.0) | **50.7%** accuracy · ECE **0.245** · p50 **13.1 ms** |
| Short 1-question payloads | typically **~10–30 ms** |
| Large state / many questions | typically **~100–160 ms** |
| vs greedy AR JSON (same weights) | roughly **10–50×** faster option scoring |

Uniform-random on the JevBench set is ~**32%** (many 4–5-way items), not 50%.
Weak gyms on full `test`: game2048 ~35%, sokoban ~38%, chess ~42%.

Example rollouts (illustration only): [`benchmarks/demos/spatial_v2_s1/`](benchmarks/demos/spatial_v2_s1/).

Raw reports: [`benchmarks/phase2_heldout__action_v2_qwen.json`](benchmarks/phase2_heldout__action_v2_qwen.json),
[`benchmarks/jevbench_action_v2_qwen.json`](benchmarks/jevbench_action_v2_qwen.json),
[`benchmarks/latency_vs_ar.json`](benchmarks/latency_vs_ar.json).  
Postmortem: [`docs/NOTE_ACTION_V2_QWEN_POSTMORTEM.md`](docs/NOTE_ACTION_V2_QWEN_POSTMORTEM.md).

---

## Limits (read this)

- **0.5B.** Useful for demos and local experiments; not a production decision model.
- **Calibration is mediocre.** Do not treat returned probabilities as
  well-calibrated confidence.
- **Different from production Jev.** Own weights, own scoring path, own
  `confidence` formula `(p_max - 1/n)/(1 - 1/n)`.
- **Closed-option scoring.** The model does not generate free text; it ranks the
  given criteria / yes–no / score symbols (usually one vocab id each). JSON option
  *keys* like `"billing"` are mapped after scoring — they are not scored as full
  strings.
- **Eval:** use the Hub dataset **`test`** split for held-out numbers — never
  score on `train`.

Training data:
[`dwidlee/systemone-lite-phase2`](https://huggingface.co/datasets/dwidlee/systemone-lite-phase2)
(240 800 train / 4 700 test).

---

## Optional: demos & retrain

```bash
pytest
python scripts/demo_systemone.py
```

Publish weights to the **stable** Hub id only:

```bash
python scripts/upload_model_hf.py --dir checkpoints/<your-run>
# → always dwidlee/systemone-lite-0.5b
```

Retrain / rebuild docs: [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## License

MIT

## Acknowledgments

System One / Jev concepts and public API docs belong to their respective owners
(TypeSafe AI). This repo is an unofficial local approximation for experimentation.
