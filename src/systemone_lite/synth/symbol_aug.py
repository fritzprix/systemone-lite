"""Glyph / legend remapping for 2D text-map gyms.

Keeps a majority of samples on canonical symbols (#/@/G/…) so demos and
fixed-legend evals stay in-distribution, while remapped samples force the
model to read the legend instead of memorizing glyphs.
"""

from __future__ import annotations

import random
from typing import Mapping

# Avoid lookalikes of each other where possible; single-width ASCII only.
_POOL = list("ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789%=&~?")

GRIDWORLD_CANONICAL: dict[str, str] = {
    "wall": "#",
    "player": "@",
    "goal": "G",
    "hazard": "X",
    "floor": ".",
}

GRIDWORLD_LABELS: dict[str, str] = {
    "wall": "Wall",
    "player": "Player",
    "goal": "Goal",
    "hazard": "Deadly Trap",
    "floor": "Floor",
}

SOKOBAN_CANONICAL: dict[str, str] = {
    "wall": "#",
    "player": "@",
    "box": "$",
    "target": ".",
    "box_on_target": "*",
    "player_on_target": "+",
    "floor": " ",
}

SOKOBAN_LABELS: dict[str, str] = {
    "wall": "Wall",
    "player": "Player",
    "box": "Box",
    "target": "Target",
    "box_on_target": "Box on target",
    "player_on_target": "Player on target",
    "floor": "Empty floor",
}


def sample_role_symbols(
    rng: random.Random,
    canonical: Mapping[str, str],
    *,
    keep_canonical_prob: float = 0.6,
) -> dict[str, str]:
    """Return role → glyph. With prob keep_canonical_prob, identity map."""
    if rng.random() < keep_canonical_prob:
        return dict(canonical)

    pool = [c for c in _POOL if c not in canonical.values()]
    # Also allow unused punctuation already in canonical set only if not chosen
    rng.shuffle(pool)
    out: dict[str, str] = {}
    used: set[str] = set()
    i = 0
    for role in canonical:
        while i < len(pool) and pool[i] in used:
            i += 1
        if i >= len(pool):
            # Fallback: keep remaining canonical
            for r2, g2 in canonical.items():
                if r2 not in out:
                    out[r2] = g2
            return out
        glyph = pool[i]
        i += 1
        out[role] = glyph
        used.add(glyph)
    return out


def role_map_to_char_map(
    canonical: Mapping[str, str], role_symbols: Mapping[str, str]
) -> dict[str, str]:
    """old glyph → new glyph (for rewriting a rendered map)."""
    return {canonical[role]: role_symbols[role] for role in canonical}


def remap_matrix(
    matrix: list[list[str]], char_map: Mapping[str, str]
) -> list[list[str]]:
    return [[char_map.get(ch, ch) for ch in row] for row in matrix]


def format_legend(role_symbols: Mapping[str, str], labels: Mapping[str, str]) -> str:
    parts = [
        f"'{role_symbols[role]}': {labels[role]}"
        for role in role_symbols
        if role in labels
    ]
    return ", ".join(parts)


def maybe_remap_gridworld_map(
    rng: random.Random,
    matrix: list[list[str]],
    *,
    keep_canonical_prob: float = 0.6,
) -> tuple[list[list[str]], str, dict[str, str]]:
    roles = sample_role_symbols(
        rng, GRIDWORLD_CANONICAL, keep_canonical_prob=keep_canonical_prob
    )
    cmap = role_map_to_char_map(GRIDWORLD_CANONICAL, roles)
    remapped = remap_matrix(matrix, cmap)
    legend = format_legend(roles, GRIDWORLD_LABELS)
    return remapped, legend, roles


def maybe_remap_sokoban_map(
    rng: random.Random,
    matrix: list[list[str]],
    *,
    keep_canonical_prob: float = 0.6,
) -> tuple[list[list[str]], str, dict[str, str]]:
    roles = sample_role_symbols(
        rng, SOKOBAN_CANONICAL, keep_canonical_prob=keep_canonical_prob
    )
    cmap = role_map_to_char_map(SOKOBAN_CANONICAL, roles)
    remapped = remap_matrix(matrix, cmap)
    legend = format_legend(roles, SOKOBAN_LABELS)
    return remapped, legend, roles
