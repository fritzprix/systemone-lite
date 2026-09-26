"""2048 sliding tile puzzle 2D text map generator & heuristic solver for System One."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any

from systemone_lite.chess_data import DistillSample, alias_criteria
from systemone_lite.synth.common import TASK_SCHEMA_ACTION_V2, choice_sample
from systemone_lite.synth.augmentation import FLIP_H_ACTION, ROT90_ACTION


def _map_action(action: str, rot_k: int, flip_h: bool) -> str:
    a = action
    for _ in range(rot_k % 4):
        a = ROT90_ACTION[a]
    if flip_h:
        a = FLIP_H_ACTION[a]
    return a


def _map_action_set(actions: list[str], rot_k: int, flip_h: bool) -> list[str]:
    return [_map_action(a, rot_k, flip_h) for a in actions]


@dataclass
class Board2048:
    grid: list[list[int]]  # 4x4 matrix of ints (0 for empty)

    def render_ascii(self) -> str:
        lines = ["+------+------+------+------+"]
        for row in self.grid:
            cells = []
            for val in row:
                if val == 0:
                    cells.append("    . ")
                else:
                    cells.append(f"{val:5d} ")
            lines.append("|" + "|".join(cells) + "|")
            lines.append("+------+------+------+------+")
        return "\n".join(lines)

    @property
    def empty_cells(self) -> list[tuple[int, int]]:
        return [(r, c) for r in range(4) for c in range(4) if self.grid[r][c] == 0]

    @property
    def max_tile(self) -> int:
        return max(max(row) for row in self.grid)

    def spawn_tile(self, rng: random.Random) -> bool:
        empty = self.empty_cells
        if not empty:
            return False
        r, c = rng.choice(empty)
        self.grid[r][c] = 4 if rng.random() < 0.1 else 2
        return True

    def slide_row_left(self, row: list[int]) -> tuple[list[int], int]:
        """Compress row to the left and merge matching tiles."""
        non_zero = [x for x in row if x != 0]
        merged = []
        score = 0
        skip = False
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                val = non_zero[i] * 2
                merged.append(val)
                score += val
                skip = True
            else:
                merged.append(non_zero[i])
        merged.extend([0] * (4 - len(merged)))
        return merged, score

    def move(self, direction: str) -> tuple[Board2048, int, bool]:
        """Apply move direction, return (new_board, score_gained, changed)."""
        new_grid = [list(r) for r in self.grid]
        total_score = 0

        if direction == "LEFT":
            for r in range(4):
                new_grid[r], s = self.slide_row_left(new_grid[r])
                total_score += s
        elif direction == "RIGHT":
            for r in range(4):
                rev, s = self.slide_row_left(new_grid[r][::-1])
                new_grid[r] = rev[::-1]
                total_score += s
        elif direction == "UP":
            for c in range(4):
                col = [new_grid[r][c] for r in range(4)]
                res, s = self.slide_row_left(col)
                for r in range(4):
                    new_grid[r][c] = res[r]
                total_score += s
        elif direction == "DOWN":
            for c in range(4):
                col = [new_grid[r][c] for r in range(4)][::-1]
                res, s = self.slide_row_left(col)
                rev = res[::-1]
                for r in range(4):
                    new_grid[r][c] = rev[r]
                total_score += s

        changed = new_grid != self.grid
        return Board2048(new_grid), total_score, changed

    def get_legal_moves(self) -> dict[str, Board2048]:
        legal = {}
        for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
            nxt, _, changed = self.move(d)
            if changed:
                legal[d] = nxt
        return legal


def score_heuristic_2048(board: Board2048) -> float:
    """Corner and monotonicity heuristic for 2048."""
    if not board.empty_cells:
        if not board.get_legal_moves():
            return -1e9  # Game over

    score = 0.0
    # Empty cells bonus (vital for survival)
    score += len(board.empty_cells) * 250.0

    # Max tile corner bonus (prefer bottom-right or bottom-left)
    max_val = board.max_tile
    corners = [board.grid[0][0], board.grid[0][3], board.grid[3][0], board.grid[3][3]]
    if max_val in corners:
        score += 1500.0

    # Monotonicity penalty: encourage monotonic row and col values
    for r in range(4):
        for c in range(3):
            val1 = board.grid[r][c]
            val2 = board.grid[r][c + 1]
            if val1 > 0 and val2 > 0:
                score -= abs(math.log2(val1) - math.log2(val2)) * 40.0
    for c in range(4):
        for r in range(3):
            val1 = board.grid[r][c]
            val2 = board.grid[r + 1][c]
            if val1 > 0 and val2 > 0:
                score -= abs(math.log2(val1) - math.log2(val2)) * 40.0

    return score


def best_move_2048(board: Board2048) -> tuple[str, float]:
    """1-ply expectimax heuristic choice for 2048."""
    legal = board.get_legal_moves()
    if not legal:
        return "UP", -1e9

    best_dir = "UP"
    best_score = -1e9
    for d, nxt in legal.items():
        val = score_heuristic_2048(nxt)
        if val > best_score:
            best_score = val
            best_dir = d
    return best_dir, best_score


def make_near_full_board(rng: random.Random) -> Board2048:
    """Synthesize a board with ≤2 empty cells for overflow-alert yes labels."""
    vals = [2, 4, 8, 16, 32, 64]
    grid = [[rng.choice(vals) for _ in range(4)] for _ in range(4)]
    n_empty = rng.randint(0, 2)
    cells = [(r, c) for r in range(4) for c in range(4)]
    for r, c in rng.sample(cells, n_empty):
        grid[r][c] = 0
    return Board2048(grid=grid)


def _overflow_alert(
    state: dict[str, Any],
    is_danger: bool,
    *,
    extra_meta: dict[str, Any] | None = None,
) -> DistillSample:
    criteria, alias_map = alias_criteria(
        {
            "yes": "Grid almost full (danger of game over)",
            "no": "Grid has sufficient space",
        }
    )
    key_to_alias = {v: k for k, v in alias_map.items()}
    lbl_key = "yes" if is_danger else "no"
    meta: dict[str, Any] = {
        "gym": "game2048",
        "schema": "noul",
        "task_schema": TASK_SCHEMA_ACTION_V2,
    }
    if extra_meta:
        meta.update(extra_meta)
    return DistillSample(
        task="game2048.overflow_alert",
        state=state,
        instructions="Is the 2048 board currently in immediate danger of overflowing and game over?",
        criteria=criteria,
        label_alias=key_to_alias[lbl_key],
        label_key=lbl_key,
        meta=meta,
    )


def generate_2048_samples(
    n_samples: int,
    *,
    seed: int = 42,
    hard: bool = False,
) -> list[DistillSample]:
    """2048 distill with balanced overflow alerts; no empty-count leak in state."""
    rng = random.Random(seed)
    samples: list[DistillSample] = []
    alert_yes = 0
    alert_no = 0
    yes_budget = max(1, n_samples // 3)

    while len(samples) < n_samples:
        # Top-up near-full boards for overflow-yes (no numeric empty_count in state)
        if alert_yes < yes_budget and alert_yes <= alert_no and rng.random() < 0.4:
            board = make_near_full_board(rng)
            from systemone_lite.synth.augmentation import apply_d4_transform

            rot_k = rng.randint(0, 3)
            flip_h = rng.random() < 0.5
            aug_grid, _ = apply_d4_transform(board.grid, None, rot_k=rot_k, flip_h=flip_h)
            aug_board = Board2048(grid=aug_grid)
            empty_count = len(aug_board.empty_cells)
            state = {
                "grid_map": f"\n{aug_board.render_ascii()}\n",
                "legend": "Numbers represent tile values; '.' represents an empty cell.",
                "highest_tile": aug_board.max_tile,
            }
            samples.append(
                _overflow_alert(
                    state,
                    empty_count <= 2,
                    extra_meta={
                        "overflow_synth": True,
                        "aug_rot_k": rot_k,
                        "aug_flip_h": flip_h,
                        "empty_cells_count": empty_count,
                    },
                )
            )
            if empty_count <= 2:
                alert_yes += 1
            else:
                alert_no += 1
            continue

        board = Board2048([[0] * 4 for _ in range(4)])
        board.spawn_tile(rng)
        board.spawn_tile(rng)

        steps = 0
        max_steps = 150

        while steps < max_steps and len(samples) < n_samples:
            legal = board.get_legal_moves()
            if not legal:
                break

            best_d, _ = best_move_2048(board)
            from systemone_lite.synth.augmentation import apply_d4_transform

            rot_k = rng.randint(0, 3)
            flip_h = rng.random() < 0.5
            aug_grid, aug_best_d = apply_d4_transform(
                board.grid, best_d, rot_k=rot_k, flip_h=flip_h
            )
            aug_board = Board2048(grid=aug_grid)
            empty_count = len(aug_board.empty_cells)

            # Keep empty count out of model-visible state (was a direct alert leak)
            state = {
                "grid_map": f"\n{aug_board.render_ascii()}\n",
                "legend": "Numbers represent tile values; '.' represents an empty cell.",
                "highest_tile": aug_board.max_tile,
            }
            aug_legal = set(
                _map_action_set(list(legal.keys()), rot_k, flip_h)
            )
            if aug_best_d is not None:
                aug_legal.add(aug_best_d)
            options = {d: f"Slide {d}" for d in sorted(aug_legal)}

            if rng.random() < 0.55 and aug_best_d is not None and aug_best_d in options:
                s = choice_sample(
                    task="game2048.slide",
                    state=state,
                    instructions=(
                        "Inspect the 4x4 2048 grid. "
                        "Choose among the listed legal slide directions."
                    ),
                    options=options,
                    label_key=aug_best_d,
                    meta={
                        "gym": "game2048",
                        "max_tile": aug_board.max_tile,
                        "aug_rot_k": rot_k,
                        "aug_flip_h": flip_h,
                        "empty_cells_count": empty_count,
                        "task_schema": TASK_SCHEMA_ACTION_V2,
                        "legal_dirs": sorted(aug_legal),
                    },
                    rng=rng,
                    hard=hard,
                )
                samples.append(s)

            is_danger = empty_count <= 2
            if is_danger and alert_yes <= alert_no and len(samples) < n_samples:
                samples.append(
                    _overflow_alert(
                        state,
                        True,
                        extra_meta={"empty_cells_count": empty_count},
                    )
                )
                alert_yes += 1
            elif (
                (not is_danger)
                and alert_no <= alert_yes
                and len(samples) < n_samples
                and rng.random() < 0.50
            ):
                samples.append(
                    _overflow_alert(
                        state,
                        False,
                        extra_meta={"empty_cells_count": empty_count},
                    )
                )
                alert_no += 1

            nxt_board = legal[best_d]
            nxt_board.spawn_tile(rng)
            board = nxt_board
            steps += 1

    return samples[:n_samples]
