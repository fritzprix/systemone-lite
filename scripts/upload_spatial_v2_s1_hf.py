#!/usr/bin/env python3
"""Upload spatial-v2-s1 continual checkpoint to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "checkpoints" / "systemone-spatial-v2-s1"
DEFAULT_REPO = "dwidlee/systemone-lite-spatial-v2-s1"

MODEL_CARD = """---
license: apache-2.0
library_name: transformers
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
  - system-one
  - decision
  - qwen2.5
  - spatial
  - cloze
datasets:
  - dwidlee/systemone-lite-phase2
---

# systemone-lite-spatial-v2-s1

Continual SFT from [`systemone-spatial-v2`](https://github.com/fritzprix/systemone-lite)
on the **zero-leakage** Phase 2 mix (CA + word games + WikiText cloze).

Not affiliated with TypeSafe AI or Jev.

| Item | Value |
|---|---|
| Init | local `checkpoints/systemone-spatial-v2` (51.2k cold-start) |
| Continue | **20 000** steps · batch 4 · lr 2e-5 · stratified · max_len 768 |
| Data | `dwidlee/systemone-lite-phase2` (train 240 800 / test 4 700; **0.00%** train∩test) |
| Wandb | [spatial-v2-s1-cloze](https://wandb.ai/doodream/systemone-lite/runs/bs62qs8x) |

## JevBench (local T=1.0, 231 tasks)

| Model | Acc | ECE |
|---|---:|---:|
| Phase 1 mixed | 45.9% | **0.221** |
| Spatial v2 | 42.9% | 0.358 |
| **This model** | **49.8%** | 0.307 |

Report in repo: `benchmarks/jevbench_spatial_v2_s1.json`.  
Not an official JevBench leaderboard submission.

## Inference

```bash
systemone-lite --model dwidlee/systemone-lite-spatial-v2-s1 --port 8000
```

## Limits

- 20k continue ≈ ⅓ epoch of the 240.8k mix (full epoch ≈ 60.2k steps).
- Held-out top-1 ≠ long rollout skill; bare demos still fail GridWorld/Sokoban clear.
- Dataset hygiene matters: use Hub `test` split only for eval — see dataset card.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if not (args.dir / "model.safetensors").exists() and not any(
        args.dir.glob("model*.safetensors")
    ):
        raise SystemExit(f"no weights under {args.dir}")

    create_repo(args.repo, private=args.private, exist_ok=True, repo_type="model")
    card = args.dir / "README.md"
    card.write_text(MODEL_CARD, encoding="utf-8")

    api = HfApi()
    api.upload_folder(
        folder_path=str(args.dir),
        repo_id=args.repo,
        repo_type="model",
        ignore_patterns=["*.tmp", ".git*", "last", "step-*", "train_meta.json"],
        commit_message="Upload spatial-v2-s1 (20k cloze continue, zero-leakage data)",
    )
    print(f"pushed {args.repo} from {args.dir}")


if __name__ == "__main__":
    main()
