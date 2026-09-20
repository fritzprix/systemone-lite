"""Invariance smoke test: verify that D4/mirror transforms commute with environment physics."""

from __future__ import annotations

import pytest
from systemone_lite.synth.augmentation import apply_connect4_mirror, apply_d4_transform
from systemone_lite.synth.connect4 import Connect4Board
from systemone_lite.synth.game2048 import Board2048
from systemone_lite.synth.gridworld import DIRECTIONS, GridWorldMap
from systemone_lite.synth.sokoban import SokobanLevel


def test_2048_d4_commutation():
    # Board with 2 at (1, 1) and 2 at (1, 2)
    grid = [
        [0, 0, 0, 0],
        [0, 2, 2, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    b = Board2048(grid=grid)
    # Moving LEFT merges the two 2s into a 4 at (1, 0)
    nxt, gained, _ = b.move("LEFT")
    assert gained == 4
    assert nxt.grid[1][0] == 4

    # Rotate original board 90 deg CW: LEFT action rotates to UP
    rot_grid, rot_act = apply_d4_transform(grid, "LEFT", rot_k=1, flip_h=False)
    assert rot_act == "UP"
    rot_b = Board2048(grid=rot_grid)
    rot_nxt, rot_gained, _ = rot_b.move(rot_act)
    assert rot_gained == 4

    # Expected: 4 at (0, 2) after 90 deg CW rotation
    assert rot_nxt.grid[0][2] == 4


def test_gridworld_d4_commutation():
    # Grid 5x5: Player at (2, 2) moves RIGHT to (2, 3)
    grid = [
        "#####",
        "#...#",
        "#.@G#",
        "#...#",
        "#####",
    ]
    matrix = [list(r) for r in grid]
    # Rotate 90 deg CW: RIGHT action maps to DOWN
    rot_m, rot_act = apply_d4_transform(matrix, "RIGHT", rot_k=1, flip_h=False)
    assert rot_act == "DOWN"
    # Player in rotated matrix is at (2, 5 - 1 - 2) = (2, 2), moving DOWN moves to (3, 2)
    # Goal G was at (2, 3), after 90 deg CW is at (3, 5 - 1 - 2) = (3, 2)
    assert rot_m[3][2] == "G"


def test_connect4_mirror_commutation():
    b = Connect4Board(grid=[["." for _ in range(7)] for _ in range(6)])
    # Drop Red in Col 2 (0-indexed 1)
    b.drop_piece(1, "R")
    assert b.grid[5][1] == "R"

    # Mirror board and action
    mir_grid, mir_col_1based = apply_connect4_mirror(b.grid, col_1based=2)
    assert mir_col_1based == 6  # 8 - 2 = 6 (0-indexed 5)
    assert mir_grid[5][5] == "R"
