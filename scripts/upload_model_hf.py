#!/usr/bin/env python3
"""Upload a local System One SFT checkpoint to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "checkpoints" / "systemone-sft"
DEFAULT_REPO = "dwidlee/systemone-lite-0.5b"

MODEL_CARD = """---
license: apache-2.0
library_name: transformers
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
  - system-one
  - decision
  - qwen2.5
datasets:
  - dwidlee/systemone-lite-general
---

# systemone-lite-0.5b

Fine-tuned weights for [`systemone-lite`](https://github.com/fritzprix/systemone-lite):
typed decisions via next-token scoring over option aliases
(`choice` / encoded `noul` / `score`).

Not affiliated with TypeSafe AI or Jev.

| Item | Value |
|---|---|
| Base | `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0) |
| Data | [`dwidlee/systemone-lite-general`](https://huggingface.co/datasets/dwidlee/systemone-lite-general) |
| Train size | 32 400 rows (ticket / alloc / debate) |
| Recipe | 1 epoch, batch size 3, gym-stratified batches, 10 800 steps |
| Loss | Cross-entropy on the labeled option alias token |

Chess-specialized weights are a **separate** local checkpoint; this model is not
trained on chess.

## Visual Demos

| Autonomous Snake Game (<12ms per step) | Tactical Chess Player (2D spatial context) |
| :---: | :---: |
| ![Snake Demo](https://raw.githubusercontent.com/fritzprix/systemone-lite/main/benchmarks/viral/snake_demo.gif) | ![Chess Demo](https://raw.githubusercontent.com/fritzprix/systemone-lite/main/benchmarks/viral/chess_proper.gif) |
| *0.5B causal LM navigating 2D grid in real-time* | *8x8 ASCII board map + debiased candidate ranking* |

## Accuracy (option top-1)

| Split | n | Base 0.5B | This model | Δ |
|---|---:|---:|---:|---:|
| iid (`test`) | 3600 | 0.439 | 0.679 | +0.240 |
| hard (`test_hard`) | 5400 | 0.427 | 0.652 | +0.225 |

Hard: alternate state layouts, option subsets, paraphrases (same label rules).
Repo reports: `benchmarks/general_*_eval*.json`.

Selected iid (this model): `ticket.route` 1.000, `alloc.fund_next` 0.980,
`ticket.needs_human` 0.775. Near base: `debate.winner` 0.493,
`debate.enough_evidence` 0.460.

### Chess representation & transfer

In chess, replacing raw FEN with an explicit 2D ASCII board map (`board_2d_map`) and debiased candidate shuffling restores genuine spatial reasoning. On the debiased benchmark, baseline zero-shot accuracy is ~6.0% (random choice among ~20 legal moves), and general SFT does not transfer to chess.

## Latency (inference path; base 0.5B measured)

In-process on **RTX 3060**, warmup excluded. Same scoring path this checkpoint
uses (0.5B forward dominates latency).

**Option scoring** vs **AR JSON** (`model.generate` greedy multi-field JSON,
full vocab). Source: repo `benchmarks/latency_vs_ar.json`.

| Case | Option p50 (ms) | AR JSON p50 (ms) | AR / option |
|---|---:|---:|---:|
| short_3q | 26.2 | 1057 | 40.3× |
| short_13q | 64.9 | 3482 | 53.7× |
| long_3q (~6k chars) | 107.6 | 1137 | 10.6× |
| long_13q | 157.9 | 3613 | 22.9× |

Option path: batched next-token logits; softmax over option token ids; prefix KV.
AR runs often used the full `max_new_tokens` budget (no early EOS). Option path
is schema-constrained; AR JSON validity is not guaranteed in this bench.

TypeSafe public materials cite Jev E2E roughly **70–500 ms** (cloud + network;
not measured here).

## Inference method (server)

1. Encode shared `state` once (prefix KV).  
2. Batch per-question suffixes.  
3. Softmax only over criteria / yes–no / score-level token ids.  
4. Assemble System One–shaped `answers`.

```bash
systemone-lite --model dwidlee/systemone-lite-0.5b --port 8000
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "dwidlee/systemone-lite-0.5b"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
```

## Limits

- Synthetic rule labels; not human preference data or RLCD.  
- Option softmax ≠ population calibration (no ECE curves published).  
- Multi-token option strings are not first-class (training uses letter aliases).
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="Upload README.md only (skip weight files)",
    )
    args = parser.parse_args()

    if not args.card_only and not (args.dir / "model.safetensors").exists() and not any(
        args.dir.glob("model*.safetensors")
    ):
        raise SystemExit(f"no weights under {args.dir}")

    create_repo(args.repo, private=args.private, exist_ok=True, repo_type="model")
    card = args.dir / "README.md"
    card.write_text(MODEL_CARD, encoding="utf-8")

    api = HfApi()
    if args.card_only:
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="model",
        )
    else:
        api.upload_folder(
            folder_path=str(args.dir),
            repo_id=args.repo,
            repo_type="model",
            ignore_patterns=["*.tmp", ".git*"],
        )
    print(f"pushed {args.repo} from {args.dir}" + (" (card only)" if args.card_only else ""))


if __name__ == "__main__":
    main()
