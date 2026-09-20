"""Connect Four 7x6 board 2D text map generator & tactical solver for System One."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

from systemone_lite.chess_data import DistillSample, alias_criteria
from systemone_lite.synth.common import choice_sample

COLS = 7
ROWS = 6


@dataclass
class Connect4Board:
    grid: list[list[str]]  # 6 rows x 7 cols, "R", "Y", or "."
    turn: str = "R"  # "R" or "Y"

    def render_ascii(self) -> str:
        lines = ["  1 2 3 4 5 6 7"]
        lines.append("+---------------+ +")
        for r in range(ROWS):
            row_str = " ".join(self.grid[r])
            lines.append(f"| {row_str} |")
        lines.append("+===============+")
        return "\n".join(lines)

    def get_legal_columns(self) -> list[int]:
        """Legal columns (0 to 6) where top row is empty."""
        return [c for c in range(COLS) if self.grid[0][c] == "."]

    def drop_piece(self, col: int, piece: str) -> int | None:
        """Drops piece into column, falls to lowest empty row. Returns row index or None."""
        if self.grid[0][col] != ".":
            return None
        for r in range(ROWS - 1, -1, -1):
            if self.grid[r][col] == ".":
                self.grid[r][col] = piece
                return r
        return None

    def check_win_at(self, r: int, c: int, piece: str) -> bool:
        """Check if placing piece at (r, c) created 4-in-a-row."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # forward
            nr, nc = r + dr, c + dc
            while 0 <= nr < ROWS and 0 <= nc < COLS and self.grid[nr][nc] == piece:
                count += 1
                nr += dr
                nc += dc
            # backward
            nr, nc = r - dr, c - dc
            while 0 <= nr < ROWS and 0 <= nc < COLS and self.grid[nr][nc] == piece:
                count += 1
                nr -= dr
                nc -= dc
            if count >= 4:
                return True
        return False

    def is_winning_move(self, col: int, piece: str) -> bool:
        legal = self.get_legal_columns()
        if col not in legal:
            return False
        # Test drop
        test_board = Connect4Board([list(row) for row in self.grid], self.turn)
        r = test_board.drop_piece(col, piece)
        if r is not None and test_board.check_win_at(r, col, piece):
            return True
        return False

    def find_immediate_threat(self, opponent_piece: str) -> int | None:
        """Returns column opponent can win with next turn, if any."""
        for c in self.get_legal_columns():
            if self.is_winning_move(c, opponent_piece):
                return c
        return None


def score_connect4_position(board: Connect4Board, player: str) -> int:
    """Tactical position evaluator for Connect Four."""
    opp = "Y" if player == "R" else "R"
    score = 0
    # Center column preference
    for r in range(ROWS):
        if board.grid[r][3] == player:
            score += 3
        elif board.grid[r][3] == opp:
            score -= 3
    return score


def best_move_connect4(board: Connect4Board, player: str) -> int:
    """Fast tactical move selector (immediate win > block opponent win > center)."""
    legal = board.get_legal_columns()
    if not legal:
        return 3

    opp = "Y" if player == "R" else "R"

    # 1. Winning move
    for c in legal:
        if board.is_winning_move(c, player):
            return c

    # 2. Block opponent's immediate winning move
    block_col = board.find_immediate_threat(opp)
    if block_col is not None and block_col in legal:
        return block_col

    # 3. Prefer central columns and avoid giving opponent a win
    scored_moves = []
    for c in legal:
        # Don't drop if it allows opponent to win directly above it!
        test_board = Connect4Board([list(row) for row in board.grid], board.turn)
        r = test_board.drop_piece(c, player)
        if r is not None and r > 0 and test_board.is_winning_move(c, opp):
            blunder_penalty = -1000
        else:
            blunder_penalty = 0

        # Center distance bonus
        center_bonus = 4 - abs(3 - c)
        scored_moves.append((c, center_bonus * 10 + blunder_penalty))

    scored_moves.sort(key=lambda x: x[1], reverse=True)
    return scored_moves[0][0]


def generate_connect4_samples(
    n_samples: int,
    *,
    seed: int = 42,
    hard: bool = False,
) -> list[DistillSample]:
    """Generate synthetic supervised choice/noul/score samples for Connect Four."""
    rng = random.Random(seed)
    samples: list[DistillSample] = []

    while len(samples) < n_samples:
        board = Connect4Board([["." for _ in range(COLS)] for _ in range(ROWS)], turn="R")
        max_turns = rng.randint(4, 25)

        for turn_idx in range(max_turns):
            if len(samples) >= n_samples:
                break
            legal = board.get_legal_columns()
            if not legal:
                break

            curr_player = "R" if turn_idx % 2 == 0 else "Y"
            opp_player = "Y" if curr_player == "R" else "R"
            best_col = best_move_connect4(board, curr_player)

            # Apply horizontal mirror reflection (gravity invariant)
            from systemone_lite.synth.augmentation import apply_connect4_mirror
            mirror_h = rng.random() < 0.5
            if mirror_h:
                aug_grid, aug_best_col_1based = apply_connect4_mirror(board.grid, best_col + 1)
                aug_board = Connect4Board(grid=aug_grid)
            else:
                aug_board = board
                aug_best_col_1based = best_col + 1

            options = {str(c + 1): f"Drop in Column {c + 1}" for c in range(COLS)}

            state = {
                "grid_map": f"\n{aug_board.render_ascii()}\n",
                "turn": f"{'Red (R)' if curr_player == 'R' else 'Yellow (Y)'}",
                "legend": "'R': Red disc, 'Y': Yellow disc, '.': Empty slot. Columns are 1 to 7.",
            }

            # 1. Choice sample: column selection
            if rng.random() < 0.70:
                s = choice_sample(
                    task="connect4.drop",
                    state=state,
                    instructions="Inspect the 7x6 Connect Four board. Choose the column (1 to 7) to drop your disc.",
                    options=options,
                    label_key=str(aug_best_col_1based),
                    meta={
                        "gym": "connect4",
                        "turn": curr_player,
                        "aug_mirror_h": mirror_h,
                    },
                    rng=rng,
                    hard=hard,
                )
                samples.append(s)

            # 2. Noul sample: opponent winning threat
            if len(samples) < n_samples and rng.random() < 0.40:
                threat_col = aug_board.find_immediate_threat(opp_player)
                has_threat = threat_col is not None
                criteria, alias_map = alias_criteria({
                    "yes": "Opponent has a 4-in-a-row threat next turn",
                    "no": "No immediate winning threat from opponent",
                })
                key_to_alias = {v: k for k, v in alias_map.items()}
                lbl_key = "yes" if has_threat else "no"
                samples.append(
                    DistillSample(
                        task="connect4.threat_alert",
                        state=state,
                        instructions="Does the opponent have an immediate 4-in-a-row winning threat next turn?",
                        criteria=criteria,
                        label_alias=key_to_alias[lbl_key],
                        label_key=lbl_key,
                        meta={"gym": "connect4", "schema": "noul"},
                    )
                )

            # Drop move
            r = board.drop_piece(best_col, curr_player)
            if r is not None and board.check_win_at(r, best_col, curr_player):
                break

    return samples[:n_samples]
