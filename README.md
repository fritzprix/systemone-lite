# systemone-lite

Local approximation of the public TypeSafe **System One** JSON API
(`POST /v1/systemone`), backed by a small causal LM.

Not affiliated with [TypeSafe AI](https://typesafe.ai) or Jev. Not a production
replacement.

**Inference:** shared state prefix (KV cache) + batched per-question next-token
scoring over **option token ids only** (softmax restricted to the question’s
criteria / yes–no / score levels). No autoregressive JSON string generation.

**Default weights:** `Qwen/Qwen2.5-0.5B-Instruct`. Optional SFT checkpoint
(mixed gyms + chess, from base):
[`dwidlee/systemone-lite-0.5b`](https://huggingface.co/dwidlee/systemone-lite-0.5b).

## Key Capabilities & Design Principles

* **⚡ Ultra-Low Latency (~10ms–35ms)**: Evaluates multiple structured questions simultaneously via next-token option-restricted softmax with shared KV cache prefix, achieving **17× to 35× speedup** over autoregressive JSON generation.
* **🎯 Pure Unbiased Evaluation**: No keyword prompt hacks (`RECOMMENDED`, `OPTIMAL`) or heuristic option order shortcuts. All evaluations rely on raw state representation and neutral option sets.
* **🗺️ 2D Spatial Environments**: Includes procedural Sokoban, 2048, GridWorld, Connect Four, and Chess environments with synthetic trajectory generation for Phase 2 spatial representation learning.

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

### Mixed SFT accuracy (option top-1)

Cold start from `Qwen/Qwen2.5-0.5B-Instruct`. Train set:
`data/mixed_train.jsonl` (43 200 rows) — ticket / alloc / debate / chess with
**equal gym exposure** (10 800 each; chess upsampled from 4 500 move rows with
`board_2d_map`). Recipe: 1 epoch, batch size 4, `--stratified`, 10 800 steps.
Checkpoint: `checkpoints/systemone-mixed-sft` → published as
[`dwidlee/systemone-lite-0.5b`](https://huggingface.co/dwidlee/systemone-lite-0.5b).

Report: [`benchmarks/mixed_vs_base_report.json`](benchmarks/mixed_vs_base_report.json).  
Write-up: [`docs/NOTE_MIXED_SFT_2026-09-20.md`](docs/NOTE_MIXED_SFT_2026-09-20.md).

#### General held-out

| Split | n | Base 0.5B | Mixed SFT | Δ |
|---|---:|---:|---:|---:|
| iid (`general_eval`) | 3600 | 0.439 | **0.781** | +0.343 |
| hard (`general_eval_hard`) | 5400 | 0.427 | **0.733** | +0.307 |

Selected iid (mixed): `ticket.route` 1.000, `ticket.urgency` 0.965,
`ticket.needs_human` 0.985, `alloc.fund_next` 0.975, `debate.winner` 0.810.
Weaker: `debate.enough_evidence` 0.550, `debate.confidence` 0.473.

(Prior general-only SFT on the same splits: iid 0.679 / hard 0.652.)

#### Chess move (2D board + shuffled options)

Eval: `data/chess_eval_5k_2d.jsonl` (n=500). Options are letter-aliased and
**order-shuffled**; raw FEN-only / fixed-order evals overstated base accuracy.

| Checkpoint | Accuracy |
|---|---:|
| Base | 0.050 |
| Mixed SFT | **0.236** |
| Prior general-only SFT | 0.042 |
| Prior chess-only SFT (FEN-era weights) | 0.028 |

Mixed training raises chess above chance without collapsing general tasks.
Absolute chess accuracy remains modest on this debiased harness.

## Resources

| Resource | Link |
|---|---|
| Code | [github.com/fritzprix/systemone-lite](https://github.com/fritzprix/systemone-lite) |
| Dataset | [dwidlee/systemone-lite-general](https://huggingface.co/datasets/dwidlee/systemone-lite-general) |
| Mixed SFT model | [dwidlee/systemone-lite-0.5b](https://huggingface.co/dwidlee/systemone-lite-0.5b) |
| Base model | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| System One API (reference) | [docs.typesafe.ai/api.md](https://docs.typesafe.ai/api.md) |
| Design notes | [`docs/PROPOSAL.md`](docs/PROPOSAL.md) |
| Mixed SFT note | [`docs/NOTE_MIXED_SFT_2026-09-20.md`](docs/NOTE_MIXED_SFT_2026-09-20.md) |

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

### Fine-tune (mixed: general + chess)

```bash
# Refresh chess JSONL with board_2d_map (keeps Stockfish labels)
python scripts/chess_distill_dataset.py \
  --refresh-file data/chess_train_5k.jsonl \
  --out data/chess_train_5k_2d.jsonl

python scripts/build_mixed_distill.py \
  --general data/general_train.jsonl \
  --chess data/chess_train_5k_2d.jsonl \
  --out data/mixed_train.jsonl \
  --chess-target 10800

python scripts/chess_finetune.py \
  --data data/mixed_train.jsonl \
  --out checkpoints/systemone-mixed-sft \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --epochs 1 --batch-size 4 --tasks all --stratified

python scripts/general_eval.py \
  --data data/general_eval.jsonl \
  --model checkpoints/systemone-mixed-sft \
  --out benchmarks/mixed_sft_general_iid.json

python scripts/chess_eval.py \
  --data data/chess_eval_5k_2d.jsonl \
  --model checkpoints/systemone-mixed-sft --task move --limit 500

python scripts/upload_model_hf.py --dir checkpoints/systemone-mixed-sft
systemone-lite --model dwidlee/systemone-lite-0.5b --port 8000
```

Numbers: [Measured results](#mixed-sft-accuracy-option-top-1).

## Chess fine-tuning (Stockfish distill)

Chess rows use `board_2d_map` (labeled 8×8 ASCII), not FEN alone. Prefer the
**mixed** checkpoint above for joint general+chess weights. A chess-only local
run is still useful for ablation:

```bash
# Ubuntu: sudo apt install stockfish
# or:     python scripts/download_stockfish.py

pip install -e ".[chess]"

python scripts/chess_distill_dataset.py --positions 500 --movetime-ms 40

python scripts/chess_finetune.py \
  --data data/chess_distill.jsonl \
  --out checkpoints/chess-sft \
  --epochs 2 --batch-size 2 --max-steps 200

python scripts/chess_eval.py --data data/chess_eval_5k_2d.jsonl \
  --model checkpoints/chess-sft --task move --limit 500
```

```bash
systemone-lite --model checkpoints/chess-sft --port 8000
```

Labels: Stockfish when available (`/usr/games/stockfish` on Ubuntu), else a
tactical heuristic. Mixed vs base numbers:
[Measured results](#chess-move-2d-board--shuffled-options).

## Interactive Demos & Dry-Runs

System One Lite provides terminal-based interactive environments with live telemetry, ANSI rendering, and `--stub` dry-run modes (which run instantly on CPU without downloading weights):

### 1. Multi-Step Chess Player (`chess_multistep_demo.py`)
Two-stage System 1 decision pipeline (Stage 1 Piece Selection → Stage 2 Destination Selection):
```bash
# Live interactive terminal demo (real weights)
python scripts/chess_multistep_demo.py --max-plies 20

# Instant dry-run (no GPU / no weights download)
python scripts/chess_multistep_demo.py --stub
```

## Spatial 2D Game Demos & Synthetic Dataset Engine

In addition to Chess, System One Lite includes full-fledged 2D spatial text-map environments with real-time heuristic/BFS solvers and instant `--stub` execution:

### 1. Sokoban (`sokoban_demo.py`)
Warehouse box-pushing puzzle with real-time deadlock detection:
```bash
python scripts/sokoban_demo.py --stub
```

### 2. 2048 (`game2048_demo.py` / `2048_demo.py`)
4x4 sliding tile puzzle with sub-10ms corner & monotonicity reflexes:
```bash
python scripts/game2048_demo.py --stub
```

### 3. GridWorld Hazards (`gridworld_demo.py`)
Procedural maze navigation with deadly lava/spike trap avoidance:
```bash
python scripts/gridworld_demo.py --stub
```

### 4. Connect Four (`connect4_demo.py`)
7-column vertical gravity board with instant 4-in-a-row threat defense:
```bash
python scripts/connect4_demo.py --stub
```

### 5. Multi-Step Chess (`chess_multistep_demo.py`)
Two-stage System 1 decision pipeline: Stage 1 (Piece Selection) → Stage 2 (Destination Selection):
```bash
python scripts/chess_multistep_demo.py --stub --max-plies 10
```

### Synthetic Dataset Synthesis (`build_spatial_distill.py`)
Generate supervised spatial datasets across all 4 games:
```bash
python scripts/build_spatial_distill.py \
  --games sokoban,game2048,gridworld,connect4 \
  --samples-per-game 500 \
  --out data/spatial_distill.jsonl
```

## Project layout

```text
src/systemone_lite/   # schema, prompt, infer (prefix cache), API, client, synth/
tests/
scripts/              # demo, latency, train/eval, HF upload
benchmarks/           # latency + held-out JSON evaluations
openapi/
docs/PROPOSAL.md
```

## Status

Toy project. Published HF weights are the **mixed** SFT (general gyms + chess
with `board_2d_map`). Limitations: multi-token option keys, modest debiased
chess accuracy, weak debate calibration, no ECE curves yet.

## License

MIT

## Acknowledgments / disclaimer

System One / Jev concepts and public API docs belong to their respective owners
(TypeSafe AI). This repository is an independent, unofficial approximation for
local experimentation only.
