"""Train/eval fingerprint helpers for zero-contamination splits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from typing import Any, Literal

from systemone_lite.chess_data import DistillSample

FpMode = Literal["full", "state_task", "state_only"]


def fingerprint(row: dict[str, Any], *, mode: FpMode = "state_only") -> str:
    """Stable MD5 of the fields used for leakage checks."""
    st = row.get("state")
    if mode == "full":
        blob = json.dumps(
            {
                "t": row["task"],
                "s": st,
                "c": row["criteria"],
                "k": row.get("label_key"),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    elif mode == "state_task":
        blob = json.dumps({"t": row["task"], "s": st}, sort_keys=True, ensure_ascii=False)
    elif mode == "state_only":
        blob = json.dumps(st, sort_keys=True, ensure_ascii=False)
    else:
        raise ValueError(f"unknown fingerprint mode: {mode!r}")
    return hashlib.md5(blob.encode()).hexdigest()


def sample_fingerprint(sample: DistillSample, *, mode: FpMode = "state_only") -> str:
    return fingerprint(sample.to_json(), mode=mode)


def collect_fingerprints(
    rows: Iterable[dict[str, Any]],
    *,
    mode: FpMode = "state_only",
) -> set[str]:
    return {fingerprint(r, mode=mode) for r in rows}


def filter_disjoint(
    candidates: Iterable[DistillSample | dict[str, Any]],
    *,
    exclude: set[str],
    mode: FpMode = "state_only",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Keep rows whose fingerprint is not in ``exclude``."""
    out: list[dict[str, Any]] = []
    for item in candidates:
        row = item.to_json() if isinstance(item, DistillSample) else dict(item)
        fp = fingerprint(row, mode=mode)
        if fp in exclude:
            continue
        out.append(row)
        exclude.add(fp)  # also de-dupe within the kept set
        if limit is not None and len(out) >= limit:
            break
    return out


def generate_disjoint(
    factory: Callable[[int, int], list[DistillSample]],
    n: int,
    *,
    exclude: set[str],
    mode: FpMode = "state_only",
    seed: int = 0,
    batch: int = 128,
    max_batches: int = 500,
) -> list[dict[str, Any]]:
    """Call ``factory(batch, seed)`` until ``n`` rows clear the exclude set.

    ``factory(count, seed)`` must return ``count`` DistillSample rows (or fewer).
    Accepted fingerprints are added to ``exclude`` so callers share one set.
    """
    if n <= 0:
        return []
    out: list[dict[str, Any]] = []
    for bi in range(max_batches):
        need = n - len(out)
        if need <= 0:
            break
        take = max(need, min(batch, n))
        batch_seed = seed + bi * 9973
        raw = factory(take, batch_seed)
        kept = filter_disjoint(raw, exclude=exclude, mode=mode, limit=need)
        out.extend(kept)
    if len(out) < n:
        raise RuntimeError(
            f"generate_disjoint: only got {len(out)}/{n} after {max_batches} batches "
            f"(mode={mode}, exclude_size={len(exclude)})"
        )
    return out[:n]
