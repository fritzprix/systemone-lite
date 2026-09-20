"""2048 sliding tile puzzle 2D text map generator & heuristic solver for System One."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any

from systemone_lite.chess_data import DistillSample, alias_criteria
from systemone_lite.synth.common import choice_sample


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


def generate_2048_samples(
    n_samples: int,
    *,
    seed: int = 42,
    hard: bool = False,
) -> list[DistillSample]:
    """Generate synthetic supervised choice/noul/score samples for 2048."""
    rng = random.Random(seed)
    samples: list[DistillSample] = []

    while len(samples) < n_samples:
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
            grid_ascii = f"\n{board.render_ascii()}\n"
            empty_count = len(board.empty_cells)

            state = {
                "grid_map": grid_ascii,
                "legend": "Numbers represent tile values; '.' represents an empty cell.",
                "empty_cells_count": empty_count,
                "highest_tile": board.max_tile,
                "grid_fullness": "CRITICAL: Under 3 empty cells left" if empty_count <= 2 else "SAFE: Ample empty cells",
            }

            # 1. Choice sample: best slide direction
            options = {
                d: f"Slide tiles {d}"
                for d in legal.keys()
            }
            for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
                if d not in options:
                    options[d] = f"BLOCKED: No tiles can move {d}"

            if rng.random() < 0.75:
                s = choice_sample(
                    task="game2048.slide",
                    state=state,
                    instructions=(
                        "Inspect the 4x4 2048 grid. Choose the best slide direction (UP, DOWN, LEFT, RIGHT) "
                        "to merge tiles cleanly and keep high-value tiles anchored in corners."
                    ),
                    options=options,
                    label_key=best_d,
                    meta={"gym": "game2048", "max_tile": board.max_tile},
                    rng=rng,
                    hard=hard,
                )
                samples.append(s)

            # 2. Noul sample: overflow / emergency alert
            if len(samples) < n_samples and rng.random() < 0.35:
                is_danger = empty_count <= 2
                criteria, alias_map = alias_criteria({"yes": "Grid almost full (danger of game over)", "no": "Grid has sufficient space"})
                key_to_alias = {v: k for k, v in alias_map.items()}
                lbl_key = "yes" if is_danger else "no"
                samples.append(
                    DistillSample(
                        task="game2048.overflow_alert",
                        state=state,
                        instructions="Is the 2048 board currently in immediate danger of overflowing and game over?",
                        criteria=criteria,
                        label_alias=key_to_alias[lbl_key],
                        label_key=lbl_key,
                        meta={"gym": "game2048", "schema": "noul"},
                    )
                )

            # Advance game
            nxt_board = legal[best_d]
            nxt_board.spawn_tile(rng)
            board = nxt_board
            steps += 1

    return samples[:n_samples]
