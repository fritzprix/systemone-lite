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
| Base | `Qwen/Qwen2.5-0.5B-Instruct` |
| Train data | [`dwidlee/systemone-lite-phase2`](https://huggingface.co/datasets/dwidlee/systemone-lite-phase2) (240 800 / 4 700; 0% train∩test) |
| Serving | option-restricted next-token scoring (closed criteria / yes–no / score) |

This repo is the **stable name**. New training runs overwrite these weights —
do not expect a new Hub repo per experiment.

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

## Performance (local)

| | |
|---|---|
| JevBench-style public set (231 tasks, T=1.0) | **49.8%** acc · ECE 0.307 · p50 **12.6 ms** |
| Short payloads | ~10–30 ms typical (RTX 3060, in-process) |

Not an official JevBench leaderboard submission. Random baseline on this set is
~32% (many 4–5-way items), not 50%.

Reports in the GitHub repo: `benchmarks/jevbench_spatial_v2_s1.json`,
`benchmarks/latency_vs_ar.json`.

## Limits

- **0.5B** — demos / local experiments, not a production decision service.
- **Calibration is mediocre** — do not trust probabilities as calibrated confidence.
- **Not Jev** — different weights, scoring path, and `confidence` formula.
- **Closed-option scoring** — ranks given symbols (usually one vocab id each);
  JSON option keys are mapped after scoring.
- Use dataset **`test`** for held-out eval — never score on `train`.

## Links

- Code: https://github.com/fritzprix/systemone-lite
- Dataset: https://huggingface.co/datasets/dwidlee/systemone-lite-phase2
