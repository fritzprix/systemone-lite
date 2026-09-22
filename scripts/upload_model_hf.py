#!/usr/bin/env python3
"""Publish current System One weights to the stable Hub id.

Always upload to ``dwidlee/systemone-lite-0.5b`` — do not mint a new repo per run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "checkpoints" / "systemone-spatial-v2-s1"
STABLE_REPO = "dwidlee/systemone-lite-0.5b"
CARD_PATH = ROOT / "scripts" / "MODEL_CARD_systemone-lite-0.5b.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
        help="Local checkpoint directory (default: current best)",
    )
    parser.add_argument("--repo", default=STABLE_REPO)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="Upload README.md only",
    )
    args = parser.parse_args()

    if not args.card_only:
        if not (args.dir / "model.safetensors").exists() and not any(
            args.dir.glob("model*.safetensors")
        ):
            raise SystemExit(f"no weights under {args.dir}")

    create_repo(args.repo, private=args.private, exist_ok=True, repo_type="model")

    card_src = CARD_PATH if CARD_PATH.exists() else args.dir / "README.md"
    if not card_src.exists():
        raise SystemExit(f"missing model card at {card_src}")
    (args.dir / "README.md").write_text(card_src.read_text(encoding="utf-8"), encoding="utf-8")

    api = HfApi()
    if args.card_only:
        api.upload_file(
            path_or_fileobj=str(args.dir / "README.md"),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="model",
            commit_message="Update model card (stable systemone-lite-0.5b)",
        )
    else:
        api.upload_folder(
            folder_path=str(args.dir),
            repo_id=args.repo,
            repo_type="model",
            ignore_patterns=["*.tmp", ".git*", "last", "step-*", "train_meta.json"],
            commit_message="Publish current systemone-lite-0.5b weights",
        )
    print(f"pushed {args.repo} from {args.dir}" + (" (card only)" if args.card_only else ""))


if __name__ == "__main__":
    main()
