"""Dihedral D4 and Reflection symmetry augmentations for 2D spatial environments.

Mathematical symmetries:
- D4 group (8 operations): 4 rotations (0, 90, 180, 270 deg) x 2 reflections (none, horizontal flip).
  Applicable to: 2048 (4x4), GridWorld (H x W), Sokoban (H x W).
- C2 reflection (horizontal flip only):
  Applicable to: Connect Four (7 columns, gravity invariant under left-right mirror).
"""

from __future__ import annotations

import copy
from typing import Any

# Action permutations under 90-degree clockwise rotation:
# UP (-1, 0) -> RIGHT (0, 1) -> DOWN (1, 0) -> LEFT (0, -1) -> UP
ROT90_ACTION = {
    "UP": "RIGHT",
    "RIGHT": "DOWN",
    "DOWN": "LEFT",
    "LEFT": "UP",
}

# Action permutations under horizontal reflection (flip columns):
# LEFT <-> RIGHT, UP <-> UP, DOWN <-> DOWN
FLIP_H_ACTION = {
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
    "UP": "UP",
    "DOWN": "DOWN",
}


def rotate_grid_90_cw(grid: list[list[Any]]) -> list[list[Any]]:
    """Rotate a 2D matrix 90 degrees clockwise.

    (r, c) in HxW -> (c, H - 1 - r) in WxH.
    """
    h = len(grid)
    w = len(grid[0])
    return [[grid[h - 1 - r][c] for r in range(h)] for c in range(w)]


def flip_grid_horizontal(grid: list[list[Any]]) -> list[list[Any]]:
    """Flip a 2D matrix horizontally (left-right reflection)."""
    return [row[::-1] for row in grid]


def apply_d4_transform(
    grid: list[list[Any]],
    action: str | None,
    rot_k: int = 0,
    flip_h: bool = False,
) -> tuple[list[list[Any]], str | None]:
    """Apply an element of the D4 group: rot_k * 90 deg CW, then optional flip_h.

    Parameters:
      grid: 2D list of cells
      action: 'UP', 'DOWN', 'LEFT', 'RIGHT', or None
      rot_k: 0, 1, 2, 3 (number of 90-deg CW rotations)
      flip_h: whether to apply horizontal flip afterwards

    Returns:
      (transformed_grid, transformed_action)
    """
    g = [list(r) for r in grid]
    a = action

    # 1. Rotations
    rot_k = rot_k % 4
    for _ in range(rot_k):
        g = rotate_grid_90_cw(g)
        if a in ROT90_ACTION:
            a = ROT90_ACTION[a]

    # 2. Horizontal Flip
    if flip_h:
        g = flip_grid_horizontal(g)
        if a in FLIP_H_ACTION:
            a = FLIP_H_ACTION[a]

    return g, a


def apply_connect4_mirror(
    grid: list[list[str]],
    col_1based: int | None,
) -> tuple[list[list[str]], int | None]:
    """Mirror Connect Four board horizontally (gravity invariant).

    Column c in 1..7 -> 8 - c in 1..7.
    """
    g = [row[::-1] for row in grid]
    new_col = (8 - col_1based) if col_1based is not None else None
    return g, new_col
