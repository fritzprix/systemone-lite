---
license: apache-2.0
library_name: transformers
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
  - system-one
  - decision
  - qwen2.5
datasets:
  - dwidlee/systemone-lite-phase2
  - dwidlee/systemone-lite-general
---

# systemone-lite-0.5b

Current published weights for
[`systemone-lite`](https://github.com/fritzprix/systemone-lite): a local
**System One–compatible** decision model on `Qwen/Qwen2.5-0.5B-Instruct`.

Not affiliated with TypeSafe AI or Jev.

| | |
|---|---|
| Base | `Qwen/Qwen2.5-0.5B-Instruct` (cold SFT) |
| Run | `action-v2-qwen` · 10 000 steps · LR 1e-5 cosine · stratified |
| Train data | [`dwidlee/systemone-lite-phase2`](https://huggingface.co/datasets/dwidlee/systemone-lite-phase2) (240 800 / 4 700; **0%** train∩test) |
| Serving | option-restricted next-token scoring (closed criteria / yes–no / score) |

This repo is the **stable name**. New training runs overwrite these weights —
do not expect a new Hub repo per experiment.

## Training notes (this revision)

- Chess: staged piece + destination (`staged_v1`, option caps ≤8).
- Spatial gyms: `action_v2` legal-only options (Connect4 drop≤3 + win_now; alerts retained).
- Sokoban eval deadlock alerts balanced 50/50 (train-time mid-run patch; see postmortem).
- Write-up: repo `docs/NOTE_ACTION_V2_QWEN_POSTMORTEM.md`.

## Use

```bash
pip install -e ".[dev]"   # from the systemone-lite repo
systemone-lite --model dwidlee/systemone-lite-0.5b --port 8000
```

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
print(response.answers["route"].choice)
```

## Performance (local, 2026-09-26)

Alias-shuffle held-out on Hub **`test`** + public JevBench. Not an official
leaderboard submission. n=800 SE ≈ ±1.8%p; n=231 SE ≈ ±3%p — treat small deltas
as noise.

| | |
|---|---|
| Phase2 held-out (`test`, n=4700) | **61.6%** |
| First-800 protocol (n=800) | **63.9%** |
| JevBench public (231 tasks, T=1.0) | **50.7%** acc · ECE **0.245** · p50 **13.1 ms** |
| Short payloads | ~10–30 ms typical (consumer GPU, in-process) |

Uniform-random on the JevBench set is ~**32%** (many 4–5-way items), not 50%.

Per-gym on full `test` (weak → strong): game2048 34.6% · sokoban 38.2% · chess 42.0% ·
gridworld 45.6% · connect4 47.0% · CA 50.5% · debate/word ~87% · cloze 89% ·
alloc/ticket ≥98%.

Reports in the GitHub repo:

- `benchmarks/phase2_heldout__action_v2_qwen.json`
- `benchmarks/jevbench_action_v2_qwen.json`
- `benchmarks/latency_vs_ar.json` (latency methodology; older run)

## Limits

- **0.5B** — demos / local experiments, not a production decision service.
- **Calibration is mediocre** — do not trust probabilities as calibrated confidence.
- **Not Jev** — different weights, scoring path, and `confidence` formula.
- **Closed-option scoring** — ranks given symbols (usually one vocab id each);
  JSON option keys are mapped after scoring.
- Spatial planning (2048 / sokoban direction / chess piece) remains far from solved.
- Use dataset **`test`** for held-out eval — never score on `train`.

## Links

- Code: https://github.com/fritzprix/systemone-lite
- Dataset: https://huggingface.co/datasets/dwidlee/systemone-lite-phase2
- Postmortem: https://github.com/fritzprix/systemone-lite/blob/main/docs/NOTE_ACTION_V2_QWEN_POSTMORTEM.md
