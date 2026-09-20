#!/usr/bin/env python3
"""Regenerate notebooks/phase2_spatial_training_colab.ipynb for Colab T4.

Data is prebuilt + uploaded to Hugging Face; Colab only downloads and trains.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "phase2_spatial_training_colab.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + "\n" for ln in text.strip("\n").split("\n")],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [ln + "\n" for ln in text.strip("\n").split("\n")],
    }


def main() -> None:
    cells = [
        md(
            """# systemone-lite — Phase 2 training (Google Colab / T4)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fritzprix/systemone-lite/blob/main/notebooks/phase2_spatial_training_colab.ipynb)

**Data is pre-generated.** This notebook downloads
[`dwidlee/systemone-lite-phase2`](https://huggingface.co/datasets/dwidlee/systemone-lite-phase2)
(204 800 train / 3 500 test: **bare spatial + bare chess**, D₄/mirror aug with `aug_*` meta)
and fine-tunes only. Do **not** regenerate data on Colab.

| Item | Value |
|---|---|
| GPU | **T4 16GB** (Runtime → T4) |
| Dataset | `dwidlee/systemone-lite-phase2` |
| Base | `Qwen/Qwen2.5-0.5B-Instruct` or Phase 1 `dwidlee/systemone-lite-0.5b` |
| smoke | 20 000-row subset, ~few thousand steps |
| full | all 204 800 rows ≈ 51 200 steps @ batch 4 (~8–10h) — use Drive |

Rebuild dataset locally (not on Colab):

```bash
python scripts/build_phase2_distill.py
python scripts/upload_phase2_hf.py
```
"""
        ),
        md("## 1. GPU check"),
        code(
            """!nvidia-smi
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("bf16_supported:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
"""
        ),
        md("## 2. Clone repo and install"),
        code(
            """from pathlib import Path

REPO = "https://github.com/fritzprix/systemone-lite.git"
ROOT_NAME = "systemone-lite"

if Path("scripts/chess_finetune.py").exists():
    print("Already in repo:", Path(".").resolve())
elif Path(ROOT_NAME, "scripts/chess_finetune.py").exists():
    get_ipython().run_line_magic("cd", ROOT_NAME)
else:
    get_ipython().system(f"git clone --depth 1 {REPO}")
    get_ipython().run_line_magic("cd", ROOT_NAME)

get_ipython().system("pip -q install -U pip")
get_ipython().system("pip -q install -e .")
get_ipython().system("pip -q install datasets huggingface_hub accelerate")
print("cwd:", Path(".").resolve())
"""
        ),
        md(
            """## 3. Config

- `MODE="smoke"` — free Colab T4.
- `MODE="full"` — entire Hub train split; mount Drive.
"""
        ),
        code(
            """from pathlib import Path

MODE = "smoke"  # "smoke" | "full"
START_FROM = "base"  # "base" | "phase1"
HF_DATA = "dwidlee/systemone-lite-phase2"
OUT_DIR = Path("checkpoints/systemone-spatial-v2")
HF_OUT_REPO = "dwidlee/systemone-spatial-0.5b"  # change to your namespace
TRAIN_JSONL = Path("data/phase2_train_colab.jsonl")

if MODE == "smoke":
    TRAIN_LIMIT = 20_000
    BATCH_SIZE = 4
    MAX_LENGTH = 768
    EPOCHS = 1
    MAX_STEPS = None
elif MODE == "full":
    TRAIN_LIMIT = None
    BATCH_SIZE = 4
    MAX_LENGTH = 768
    EPOCHS = 1
    MAX_STEPS = None
else:
    raise ValueError(MODE)

BASE_MODEL = (
    "Qwen/Qwen2.5-0.5B-Instruct"
    if START_FROM == "base"
    else "dwidlee/systemone-lite-0.5b"
)
print(
    dict(
        MODE=MODE,
        HF_DATA=HF_DATA,
        TRAIN_LIMIT=TRAIN_LIMIT,
        BASE_MODEL=BASE_MODEL,
        BATCH_SIZE=BATCH_SIZE,
        MAX_LENGTH=MAX_LENGTH,
        OUT_DIR=str(OUT_DIR),
    )
)
"""
        ),
        md(
            """## 4. Download Hub dataset → train JSONL

Uses `scripts/hf_phase2_to_jsonl.py` (no on-Colab spatial generation).
"""
        ),
        code(
            """import json
import subprocess
from collections import Counter
from pathlib import Path

Path("data").mkdir(exist_ok=True)
cmd = [
    "python",
    "scripts/hf_phase2_to_jsonl.py",
    "--repo",
    HF_DATA,
    "--split",
    "train",
    "--out",
    str(TRAIN_JSONL),
    "--seed",
    "42",
]
if TRAIN_LIMIT:
    cmd += ["--limit", str(TRAIN_LIMIT)]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)

gyms = Counter()
n = 0
banned = ("RECOMMENDED", "BLOCKED", "CRITICAL:", "SAFE:", "Push box", "hazard_nearby")
hits = 0
with TRAIN_JSONL.open() as f:
    for line in f:
        n += 1
        if any(b in line for b in banned):
            hits += 1
        row = json.loads(line)
        gyms[(row.get("meta") or {}).get("gym", "?")] += 1
print("rows", n)
print("gyms", dict(gyms))
print("forbidden coaching hits", hits)
"""
        ),
        md("## 5. Optional Drive checkpoints (recommended for full)"),
        code(
            """USE_DRIVE = MODE == "full"

if USE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    OUT_DIR = Path("/content/drive/MyDrive/systemone-lite") / OUT_DIR.name
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    print("Checkpoints ->", OUT_DIR)
else:
    print("Checkpoints ->", OUT_DIR)
"""
        ),
        md("## 6. Train (fp16 on T4)"),
        code(
            """import math
import subprocess
from pathlib import Path

n_lines = sum(1 for _ in TRAIN_JSONL.open())
steps_est = math.ceil(n_lines / BATCH_SIZE) * EPOCHS
if MAX_STEPS:
    steps_est = min(steps_est, MAX_STEPS)
print(f"rows={n_lines} batch={BATCH_SIZE} -> ~{steps_est} steps")
print("Rough T4 wall ~0.5-0.8s/step ->", round(steps_est * 0.65 / 3600, 2), "h")

cmd = [
    "python",
    "scripts/chess_finetune.py",
    "--data",
    str(TRAIN_JSONL),
    "--out",
    str(OUT_DIR),
    "--model",
    BASE_MODEL,
    "--epochs",
    str(EPOCHS),
    "--batch-size",
    str(BATCH_SIZE),
    "--max-length",
    str(MAX_LENGTH),
    "--lr",
    "5e-5",
    "--tasks",
    "all",
    "--stratified",
]
if MAX_STEPS:
    cmd += ["--max-steps", str(MAX_STEPS)]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)
for p in sorted(Path(OUT_DIR).glob("*")):
    print(p.name, p.stat().st_size)
"""
        ),
        md("## 7. Short quiet demos (sanity only)"),
        code(
            """import subprocess

for script in [
    "scripts/game2048_demo.py",
    "scripts/sokoban_demo.py",
    "scripts/gridworld_demo.py",
]:
    cmd = ["python", script, "--model", str(OUT_DIR), "--quiet", "--max-steps", "8"]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
"""
        ),
        md("## 8. Upload checkpoint (optional)"),
        code(
            """UPLOAD = False  # set True to push

if UPLOAD:
    from huggingface_hub import HfApi, login

    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN")
    except Exception:
        import getpass

        token = getpass.getpass("HF write token: ")
    login(token=token)
    api = HfApi()
    api.create_repo(HF_OUT_REPO, exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(OUT_DIR),
        repo_id=HF_OUT_REPO,
        commit_message=f"Phase 2 Colab MODE={MODE} from {BASE_MODEL}",
    )
    print("https://huggingface.co/" + HF_OUT_REPO)
else:
    print("Skip upload. Checkpoint at", OUT_DIR)
"""
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "cells": cells,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
