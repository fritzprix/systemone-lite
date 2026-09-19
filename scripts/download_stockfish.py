#!/usr/bin/env python3
"""Download a portable Stockfish binary into third_party/."""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = (
    "https://github.com/official-stockfish/Stockfish/releases/download/"
    "sf_17/stockfish-ubuntu-x86-64-avx2.tar"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=Path,
        default=ROOT / "third_party",
    )
    args = parser.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)
    tar_path = args.dir / "stockfish.tar"
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, tar_path)
    with tarfile.open(tar_path) as tf:
        tf.extractall(args.dir)
    tar_path.unlink(missing_ok=True)
    binary = args.dir / "stockfish" / "stockfish-ubuntu-x86-64-avx2"
    binary.chmod(0o755)
    print(f"ready: {binary}")


if __name__ == "__main__":
    main()
