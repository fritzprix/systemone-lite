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

Toy **System One–style** typed-decision checkpoint for
[`systemone-lite`](https://github.com/fritzprix/systemone-lite).

- Base: `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0)
- Train data: [`dwidlee/systemone-lite-general`](https://huggingface.co/datasets/dwidlee/systemone-lite-general)
  (ticket / resource / debate gyms, letter-alias `choice` labels)
- Recipe: 2000 CE steps on option-alias next-token, batch size 2

**Not** affiliated with TypeSafe AI / Jev. Chess-specialized weights are separate
(`checkpoints/chess-sft` locally).

## Held-out accuracy (`data/general_eval.jsonl`, n=1800)

| Model | Accuracy |
|---|---:|
| Base Qwen2.5-0.5B-Instruct | 0.422 |
| This checkpoint | **0.607** |

Largest gains: `ticket.route`, `ticket.urgency`, `alloc.fund_next`.

## Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "dwidlee/systemone-lite-0.5b"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
```

Or with the project server:

```bash
systemone-lite --model dwidlee/systemone-lite-0.5b --port 8000
```
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
        ignore_patterns=["*.tmp", ".git*"],
    )
    print(f"pushed {args.repo} from {args.dir}")


if __name__ == "__main__":
    main()
