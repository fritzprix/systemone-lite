from __future__ import annotations

import pytest
from systemone_lite.synth.augmentation import (
    apply_connect4_mirror,
    apply_d4_transform,
    flip_grid_horizontal,
    rotate_grid_90_cw,
)


def test_d4_rotations_cycle():
    grid = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    # 90 deg: 2x3 -> 3x2
    g90, a90 = apply_d4_transform(grid, "UP", rot_k=1, flip_h=False)
    assert len(g90) == 3 and len(g90[0]) == 2
    assert a90 == "RIGHT"
    assert g90 == [
        [4, 1],
        [5, 2],
        [6, 3],
    ]

    # 180 deg
    g180, a180 = apply_d4_transform(grid, "UP", rot_k=2, flip_h=False)
    assert a180 == "DOWN"
    assert g180 == [
        [6, 5, 4],
        [3, 2, 1],
    ]

    # 270 deg
    g270, a270 = apply_d4_transform(grid, "UP", rot_k=3, flip_h=False)
    assert a270 == "LEFT"

    # 360 deg returns to original
    g360, a360 = apply_d4_transform(grid, "UP", rot_k=4, flip_h=False)
    assert g360 == grid
    assert a360 == "UP"


def test_d4_flip_and_composite():
    grid = [
        [1, 2],
        [3, 4],
    ]
    # Horizontal flip: LEFT <-> RIGHT
    g_flip, a_flip = apply_d4_transform(grid, "LEFT", rot_k=0, flip_h=True)
    assert a_flip == "RIGHT"
    assert g_flip == [
        [2, 1],
        [4, 3],
    ]

    # Double flip returns identity
    g_id, a_id = apply_d4_transform(g_flip, a_flip, rot_k=0, flip_h=True)
    assert g_id == grid
    assert a_id == "LEFT"


def test_connect4_mirror():
    grid = [
        [".", ".", ".", "R", ".", ".", "."],
        [".", ".", "Y", "R", ".", ".", "."],
    ]
    g_mir, col_mir = apply_connect4_mirror(grid, col_1based=1)
    assert col_mir == 7
    # Center column stays center
    _, center_mir = apply_connect4_mirror(grid, col_1based=4)
    assert center_mir == 4
    # Column 3 maps to 5
    _, col3_mir = apply_connect4_mirror(grid, col_1based=3)
    assert col3_mir == 5
    # Token positions mirrored
    assert g_mir[1][4] == "Y"  # was col index 2 (Col 3), now col index 4 (Col 5)
