#!/usr/bin/env python3
"""Upload a local System One SFT checkpoint to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "checkpoints" / "systemone-mixed-sft"
DEFAULT_REPO = "dwidlee/systemone-lite-0.5b"

MODEL_CARD = """---
license: apache-2.0
library_name: transformers
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
  - system-one
  - decision
  - qwen2.5
  - chess
datasets:
  - dwidlee/systemone-lite-general
---

# systemone-lite-0.5b

Fine-tuned weights for [`systemone-lite`](https://github.com/fritzprix/systemone-lite):
typed decisions via next-token scoring over option aliases.

Not affiliated with TypeSafe AI or Jev.

| Item | Value |
|---|---|
| Base | `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0) |
| Train data | `mixed_train.jsonl` — ticket / alloc / debate + chess |
| Mix | 43 200 rows; **10 800 per gym** (chess upsampled from 4 500) |
| Chess state | `board_2d_map` (8×8 ASCII with ranks/files), not FEN-only |
| Recipe | cold start from base; 1 epoch; batch 4; stratified; 10 800 steps |
| Loss | Cross-entropy on the labeled option alias token |
| Local path | `checkpoints/systemone-mixed-sft` |

## Accuracy vs base (option top-1)

Source: repo `benchmarks/mixed_vs_base_report.json`.

### General

| Split | n | Base | This model | Δ |
|---|---:|---:|---:|---:|
| iid | 3600 | 0.439 | **0.781** | +0.343 |
| hard | 5400 | 0.427 | **0.733** | +0.307 |

Selected iid: `ticket.route` 1.000, `ticket.urgency` 0.965,
`ticket.needs_human` 0.985, `alloc.fund_next` 0.975, `debate.winner` 0.810.

### Chess move (2D + shuffled options, n=500)

| Checkpoint | Accuracy |
|---|---:|
| Base | 0.050 |
| This model | **0.236** |

Eval uses labeled `board_2d_map` and shuffled letter aliases. Fixed-order /
FEN-only harnesses previously inflated base scores via option-order bias.

## Latency (same 0.5B inference path)

In-process **RTX 3060**, option scoring vs AR JSON
(`benchmarks/latency_vs_ar.json`):

| Case | Option p50 (ms) | AR JSON p50 (ms) | AR / option |
|---|---:|---:|---:|
| short_3q | 26.2 | 1057 | 40.3× |
| short_13q | 64.9 | 3482 | 53.7× |
| long_3q | 107.6 | 1137 | 10.6× |
| long_13q | 157.9 | 3613 | 22.9× |

## Inference

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

- Synthetic / engine labels; not human prefs or RLCD.  
- Option softmax ≠ population calibration (no ECE).  
- Chess absolute accuracy on the debiased harness is still modest.  
- Multi-token option strings are not first-class (letter aliases in training).
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
