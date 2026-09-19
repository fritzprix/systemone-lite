"""Confidence derived from a probability distribution (Choice / Score)."""

from __future__ import annotations


def distribution_confidence(probabilities: dict[str, float]) -> float:
    """Collapse a distribution into [0, 1] confidence.

    Uses: (p_max - 1/n) / (1 - 1/n), clipped to [0, 1].
    This may differ from TypeSafe's unpublished formula; probabilities remain
    the source of truth for clients that want a custom statistic.
    """
    n = len(probabilities)
    if n < 2:
        return 1.0 if n == 1 else 0.0

    p_max = max(probabilities.values())
    uniform = 1.0 / n
    denom = 1.0 - uniform
    if denom <= 0.0:
        return 0.0
    raw = (p_max - uniform) / denom
    return float(min(1.0, max(0.0, raw)))
