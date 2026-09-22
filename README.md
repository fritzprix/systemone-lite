# systemone-lite

Local **System One–shaped** decision API: you send `state` + typed questions
(`noul` / `choice` / `score`), and get back **one option letter (or yes/no/level)** —
not free-form JSON.

Built on `Qwen/Qwen2.5-0.5B-Instruct` with shared KV prefix + option-restricted
softmax. Runs on a consumer GPU (figures below: RTX 3060).

> Not affiliated with [TypeSafe AI](https://typesafe.ai) or Jev. Not a hosted product.

---

## Which weights should I use?

| Checkpoint | Hub | Use when |
|---|---|---|
| **Recommended — spatial-v2-s1** | [`dwidlee/systemone-lite-spatial-v2-s1`](https://huggingface.co/dwidlee/systemone-lite-spatial-v2-s1) | Best current all-rounder (text + spatial + cloze continue) |
| Phase 1 mixed | [`dwidlee/systemone-lite-0.5b`](https://huggingface.co/dwidlee/systemone-lite-0.5b) | Text routing / ticket-style tasks only; best calibration |
| Base | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Ablation / cold start |

Dataset for Phase 2 training/eval:
[`dwidlee/systemone-lite-phase2`](https://huggingface.co/datasets/dwidlee/systemone-lite-phase2)
(**train 240 800 / test 4 700**, **0%** train∩test state overlap).

---

## Performance (recommended model)

**`spatial-v2-s1`** — continual SFT from spatial-v2 on the scrubbed Phase 2 mix
(20 000 steps). Local benches only; **not** an official JevBench leaderboard row.

| Bench | Result |
|---|---|
| JevBench public subset (231 tasks, T=1.0) | **Acc 49.8%** · ECE 0.307 · p50 **12.6 ms** |
| Phase 1 mixed (same JevBench harness) | Acc 45.9% · ECE **0.221** |
| Spatial held-out (parent `spatial-v2`, bare+shuffled) | overall **0.53** top-1 across 4 gyms |

Latency (in-process option scoring, same machine class): short 1-question payloads
~**10–30 ms**; large state / many questions ~**100–160 ms**. Option scoring is
typically **10–50×** faster than greedy AR JSON on the same weights
([`benchmarks/latency_vs_ar.json`](benchmarks/latency_vs_ar.json)).

Bare-face rollouts (GIFs): [`benchmarks/demos/spatial_v2_s1/`](benchmarks/demos/spatial_v2_s1/).

Training / eval notes and older checkpoint tables live under [`docs/`](docs/) —
start with [`NOTE_S1_CLOZE_AND_LEAKAGE_2026-09-22.md`](docs/NOTE_S1_CLOZE_AND_LEAKAGE_2026-09-22.md).

---

## Limits

- **0.5B single-token policy** — strong at short typed decisions; weak at long
  planning (Sokoban/chess rollouts still fail often even when held-out top-1 looks OK).
- **Calibration** — v2-s1 accuracy beat Phase 1 on JevBench; ECE did **not**. Prefer
  Phase 1 mixed if you care more about confidence quality.
- **Not Jev** — different model, sampler, and `confidence` formula
  `(p_max - 1/n)/(1 - 1/n)`.
- **Eval hygiene** — always use the Hub **`test`** split (or rebuilt held-out JSONL).
  Never score on `train`. Earlier mixes leaked train states into eval; current Hub
  revision is scrubbed to **0%** overlap.
- **Multi-token option keys** are not first-class (training uses letter aliases).

---

## Install

```bash
git clone https://github.com/fritzprix/systemone-lite.git
cd systemone-lite
pip install -e ".[dev]"
```

Python 3.11+ · GPU recommended · see `pyproject.toml`.

---

## Quick start

### API server

```bash
systemone-lite --model dwidlee/systemone-lite-spatial-v2-s1 --port 8000

curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @tests/fixtures/official_example_request.json
```

### Python client

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

print(response.answers["route"].choice)
print(response.answers["needs_review"].noul)
print(response.answers["urgency"].score)
```

Wire shape matches the public System One contract
([TypeSafe API reference](https://docs.typesafe.ai/api.md)): `POST /v1/systemone`
with `noul` | `choice` | `score`. OpenAPI sketch: [`openapi/systemone.yaml`](openapi/systemone.yaml).

### Demos (optional)

```bash
pytest                                          # no weights
python scripts/demo_systemone.py                # downloads weights once
python scripts/run_bare_demos.py --only spatial_v2_s1
```

---

## What this project is

| Is | Isn’t |
|---|---|
| Local FastAPI + client approximating System One **JSON contracts** | Clone of Jev weights / RLCD / hosted SaaS |
| Option-id softmax + prefix KV (low latency) | Autoregressive JSON generation |
| Open MIT research / hobby stack | Production TypeSafe replacement |

**Why it exists:** run System One–shaped clients offline, inspect option scoring,
and fine-tune on synthetic typed-decision gyms (tickets, budgets, debates, 2D maps,
cloze).

---

## Train / rebuild (optional)

```bash
# Phase 2 mix (spatial + chess + general + CA + word + cloze)
python scripts/build_phase2_distill.py
python scripts/build_synth_diversity.py --merge-into-phase2
python scripts/audit_train_eval_overlap.py   # must be 0%
python scripts/upload_phase2_hf.py

python scripts/chess_finetune.py \
  --data data/phase2_train_200k.jsonl \
  --out checkpoints/systemone-spatial-v2-s1 \
  --model checkpoints/systemone-spatial-v2 \
  --epochs 1 --batch-size 4 --tasks all --stratified --max-steps 20000
```

Colab notebook: [`notebooks/phase2_spatial_training_colab.ipynb`](notebooks/phase2_spatial_training_colab.ipynb).

Roadmap & research logs: [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## License

MIT

## Acknowledgments

System One / Jev concepts and public API docs belong to their respective owners
(TypeSafe AI). This repository is an independent, unofficial approximation for
local experimentation only.
