# systemone-lite

Local approximation of the public TypeSafe **System One** JSON API
(`POST /v1/systemone`), backed by a small causal LM.

Not affiliated with [TypeSafe AI](https://typesafe.ai) or Jev. Not a production
replacement.

**Inference:** shared state prefix (KV cache) + batched per-question next-token
scoring over **option token ids only** (softmax restricted to the question’s
criteria / yes–no / score levels). No autoregressive JSON string generation.

**Default weights:** `Qwen/Qwen2.5-0.5B-Instruct`. Optional SFT checkpoint:
[`dwidlee/systemone-lite-0.5b`](https://huggingface.co/dwidlee/systemone-lite-0.5b).

## Measured results

All figures below are from this repo’s scripts. Reproduce paths are linked.

### API surface

Compatible with the public System One **request/response shape**
([TypeSafe API reference](https://docs.typesafe.ai/api.md)):

- `POST /v1/systemone`
- `state` + `questions` map: `noul` | `choice` | `score`
- `answers` + `usage.input_tokens` / `usage.output_tokens`

Intentionally different: model ids, accuracy/calibration, auth, and
`confidence` formula `(p_max - 1/n)/(1 - 1/n)`.

### Latency (in-process, no HTTP)

Hardware: **NVIDIA RTX 3060** (12 GB). Model: `Qwen/Qwen2.5-0.5B-Instruct`.
Warmup excluded.

**Option scoring** = System One path (batched next-token logits; softmax over
option token ids; prefix KV when ≥2 questions).  
**AR JSON** = same weights, `model.generate` greedy multi-field JSON
(full vocabulary; fixed `max_new_tokens` budget).

Sources:
[`benchmarks/latency_prefix_cache.json`](benchmarks/latency_prefix_cache.json)
(option-only sweep),
[`benchmarks/latency_vs_ar.json`](benchmarks/latency_vs_ar.json)
(option vs AR).

#### Option scoring (prefix KV)

| Case | Questions | State chars | p50 (ms) |
|---|---:|---:|---:|
| short_1q | 1 | 73 | 10.7 |
| short_3q | 3 | 73 | 24.4 |
| short_8q | 8 | 73 | 39.0 |
| short_13q | 13 | 73 | 58.4 |
| long_3q | 3 | 6274 | 98.8 |
| long_13q | 13 | 6274 | 145.0 |

#### Option scoring vs AR JSON (same machine / weights)

| Case | Option p50 (ms) | AR JSON p50 (ms) | AR / option |
|---|---:|---:|---:|
| short_3q | 26.2 | 1057 | 40.3× |
| short_13q | 64.9 | 3482 | 53.7× |
| long_3q | 107.6 | 1137 | 10.6× |
| long_13q | 157.9 | 3613 | 22.9× |

Notes: AR wall time is `generate()` only (no HTTP). In these runs AR often
filled the `max_new_tokens` budget (no early EOS), so treat ratios as an upper
bound on AR cost for that budget. Option scoring returns schema symbols by
construction; AR may emit invalid JSON (not scored here).

TypeSafe’s public materials describe Jev E2E latency roughly in the
**70–500 ms** range (their cloud + network; not measured here). Not a controlled
comparison to this local bench.

```bash
python scripts/bench_latency.py --warmup 3 --runs 15 \
  --out benchmarks/latency_vs_ar.json
```

### General SFT accuracy (option top-1)

Dataset: [`dwidlee/systemone-lite-general`](https://huggingface.co/datasets/dwidlee/systemone-lite-general)
(ticket / alloc / debate gyms). Train: 32 400 rows, 1 epoch, batch size 3,
gym-stratified batches (10 800 steps; equal gym exposure).
Checkpoint: [`dwidlee/systemone-lite-0.5b`](https://huggingface.co/dwidlee/systemone-lite-0.5b).

| Split | n | Base 0.5B | SFT | Δ |
|---|---:|---:|---:|---:|
| iid (`test`) | 3600 | 0.439 | 0.679 | +0.240 |
| hard (`test_hard`) | 5400 | 0.427 | 0.652 | +0.225 |

Hard split: alternate state layouts, option subsets, paraphrases (same label
rules). Per-task JSON:
[`benchmarks/general_base_eval.json`](benchmarks/general_base_eval.json),
[`benchmarks/general_sft_eval.json`](benchmarks/general_sft_eval.json),
[`benchmarks/general_sft_eval_hard.json`](benchmarks/general_sft_eval_hard.json).

Selected iid tasks (SFT): `ticket.route` 1.000, `alloc.fund_next` 0.980,
`ticket.needs_human` 0.775. Near-base: `debate.winner` 0.493,
`debate.enough_evidence` 0.460.

### Chess (no positive transfer from general SFT)

Move-choice top-1 on `data/chess_eval_5k.jsonl` (n=500).
Source: [`benchmarks/chess_transfer_from_general.json`](benchmarks/chess_transfer_from_general.json).

| Checkpoint | Accuracy |
|---|---:|
| Base `Qwen2.5-0.5B-Instruct` | 0.790 |
| General SFT (`systemone-lite-0.5b`) | 0.458 |
| Chess SFT (`checkpoints/chess-sft`, local) | 0.834 |

General and chess checkpoints are separate; mixing domains in one weight file
is not claimed to transfer.

## Resources

| Resource | Link |
|---|---|
| Code | [github.com/fritzprix/systemone-lite](https://github.com/fritzprix/systemone-lite) |
| Dataset | [dwidlee/systemone-lite-general](https://huggingface.co/datasets/dwidlee/systemone-lite-general) |
| General SFT model | [dwidlee/systemone-lite-0.5b](https://huggingface.co/dwidlee/systemone-lite-0.5b) |
| Base model | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| System One API (reference) | [docs.typesafe.ai/api.md](https://docs.typesafe.ai/api.md) |
| Design notes | [`docs/PROPOSAL.md`](docs/PROPOSAL.md) |

```bash
systemone-lite --model dwidlee/systemone-lite-0.5b --port 8000

python - <<'PY'
from datasets import load_dataset
print(load_dataset("dwidlee/systemone-lite-general"))
PY
```

## Why this exists

- Run System One–shaped clients against a local server (no TypeSafe account)
- Inspect batched option scoring and prefix KV behavior
- Fine-tune on synthetic typed-decision gyms

## What it is / isn’t

| Is | Isn’t |
|---|---|
| Approximation of System One **JSON contracts** | Clone of Jev weights, sampler, or RLCD |
| Local FastAPI + Python client | Hosted SaaS |
| Batched next-token + option-restricted softmax + prefix KV | Autoregressive JSON generation; BERT MLM |
| MIT open source | Affiliated with or endorsed by TypeSafe |

## Requirements

- Python 3.11+  
- `torch`, `transformers`, `fastapi` (see `pyproject.toml`)  
- GPU recommended (figures above: RTX 3060); CPU works, slower  

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

Point a System One–shaped client at `http://127.0.0.1:8000` for local experiments.

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

OpenAPI sketch: [`openapi/systemone.yaml`](openapi/systemone.yaml)

## General synthetic dataset (multi-gym)

Non-chess typed decisions (ticket routing, budget allocation, debate judging).
Rows are letter-alias `choice` samples for `scripts/chess_finetune.py`.

Build notes:

1. Round-robin gym episodes  
2. Stratified train / iid-eval split by task  
3. Separate hard eval (layout / paraphrase / option-subset shift)  
4. Train with `--stratified` so batches mix gyms  

HF mirror: **[`dwidlee/systemone-lite-general`](https://huggingface.co/datasets/dwidlee/systemone-lite-general)**  
(`train` 32.4k / `test` 3.6k / `test_hard` 5.4k / `full` 36k).  
Re-upload: `python scripts/upload_general_hf.py`.

```bash
python scripts/build_general_distill.py \
  --episodes 4000 --hard-episodes 600 --eval-frac 0.1 --seed 0

wc -l data/general_train.jsonl data/general_eval.jsonl data/general_eval_hard.jsonl
```

| Gym | Tasks |
|---|---|
| `ticket` | `ticket.route`, `ticket.needs_human`, `ticket.urgency` |
| `alloc` | `alloc.fund_next`, `alloc.can_fund_all`, `alloc.pressure` |
| `debate` | `debate.winner`, `debate.enough_evidence`, `debate.confidence` |

### Fine-tune

```bash
python scripts/chess_finetune.py \
  --data data/general_train.jsonl \
  --out checkpoints/systemone-sft \
  --epochs 1 --batch-size 3 --tasks all --stratified

python scripts/general_eval.py \
  --data data/general_eval.jsonl \
  --model checkpoints/systemone-sft \
  --out benchmarks/general_sft_eval.json

python scripts/general_eval.py \
  --data data/general_eval_hard.jsonl \
  --model checkpoints/systemone-sft \
  --out benchmarks/general_sft_eval_hard.json

python scripts/upload_model_hf.py
systemone-lite --model dwidlee/systemone-lite-0.5b --port 8000
```

Accuracy numbers: see [Measured results](#measured-results).

## Chess fine-tuning (Stockfish distill)

Separate checkpoint for legal move choice. Not the general HF model above.

```bash
# Ubuntu: sudo apt install stockfish
# or:     python scripts/download_stockfish.py

pip install -e ".[chess]"

python scripts/chess_distill_dataset.py --positions 500 --movetime-ms 40

python scripts/chess_finetune.py \
  --data data/chess_distill.jsonl \
  --out checkpoints/chess-sft \
  --epochs 2 --batch-size 2 --max-steps 200

python scripts/chess_eval.py --data data/chess_eval_5k.jsonl \
  --model checkpoints/chess-sft --task move --limit 500
```

```bash
systemone-lite --model checkpoints/chess-sft --port 8000
```

Labels: Stockfish when available (`/usr/games/stockfish` on Ubuntu), else a
tactical heuristic. Transfer numbers: [Measured results](#chess-no-positive-transfer-from-general-sft).

## Viral chess demo (local video)

```bash
pip install -e ".[viral]"

python scripts/chess_viral_demo.py --plies 20 --fps 12 \
  --model checkpoints/chess-sft
```

Outputs: `benchmarks/viral/systemone_lite_chess.mp4`, `.gif`  
Dry-run: `python scripts/chess_viral_demo.py --stub --plies 6`

## Project layout

```text
src/systemone_lite/   # schema, prompt, infer (prefix cache), API, client, synth/
tests/
scripts/              # demo, latency, train/eval, HF upload
benchmarks/           # latency + held-out JSON (+ viral clips)
openapi/
docs/PROPOSAL.md
```

## Status

Toy project. Published: general dataset + SFT on Hugging Face. Chess SFT is
local-only. Limitations: multi-token option keys, weak debate tasks on 0.5B,
no reliability diagrams / ECE yet (option softmax ≠ population calibration).

## License

MIT

## Acknowledgments / disclaimer

System One / Jev concepts and public API docs belong to their respective owners
(TypeSafe AI). This repository is an independent, unofficial approximation for
local experimentation only.
