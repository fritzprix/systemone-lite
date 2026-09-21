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


def best_move_connect4(
    board: Connect4Board,
    player: str,
    rng: random.Random | None = None,
    *,
    column_bias: dict[int, float] | None = None,
) -> int:
    """Tactical move selector (win > block > safe free move).

    Free moves prefer under-filled columns when ``column_bias`` maps
    0-based col → weight (higher = more desired). Default is uniform among
    non-blunder legal columns (no soft center prior).
    """
    legal = board.get_legal_columns()
    if not legal:
        return 3

    opp = "Y" if player == "R" else "R"
    rng = rng or random.Random()

    for c in legal:
        if board.is_winning_move(c, player):
            return c

    block_col = board.find_immediate_threat(opp)
    if block_col is not None and block_col in legal:
        return block_col

    safe: list[int] = []
    for c in legal:
        test_board = Connect4Board([list(row) for row in board.grid], board.turn)
        r = test_board.drop_piece(c, player)
        # Avoid moves that gift an immediate win to the opponent
        if r is not None and r > 0 and test_board.is_winning_move(c, opp):
            continue
        safe.append(c)
    if not safe:
        safe = list(legal)

    if column_bias:
        weights = [max(1e-3, float(column_bias.get(c, 1.0))) for c in safe]
        return rng.choices(safe, weights=weights, k=1)[0]
    return rng.choice(safe)


def _drop_column_weights(counts: dict[str, int]) -> dict[int, float]:
    """Inverse-frequency weights for 0-based columns (keys are 1-based label strings)."""
    weights: dict[int, float] = {}
    for c in range(COLS):
        n = counts.get(str(c + 1), 0)
        weights[c] = 1.0 / (1.0 + n)
    return weights


def _threat_alert(
    state: dict[str, Any],
    has_threat: bool,
    *,
    extra_meta: dict[str, Any] | None = None,
) -> DistillSample:
    criteria, alias_map = alias_criteria(
        {
            "yes": "Opponent has a 4-in-a-row threat next turn",
            "no": "No immediate winning threat from opponent",
        }
    )
    key_to_alias = {v: k for k, v in alias_map.items()}
    lbl_key = "yes" if has_threat else "no"
    meta: dict[str, Any] = {"gym": "connect4", "schema": "noul"}
    if extra_meta:
        meta.update(extra_meta)
    return DistillSample(
        task="connect4.threat_alert",
        state=state,
        instructions="Does the opponent have an immediate 4-in-a-row winning threat next turn?",
        criteria=criteria,
        label_alias=key_to_alias[lbl_key],
        label_key=lbl_key,
        meta=meta,
    )


def make_horizontal_threat_board(rng: random.Random) -> tuple[Connect4Board, str, int]:
    """Board where opponent has an immediate horizontal win threat; return (board, to_move, block_col)."""
    board = Connect4Board([["." for _ in range(COLS)] for _ in range(ROWS)], turn="R")
    # Bottom-row threat: three Y in a row, open cell on either side → R to move must block
    start = rng.randint(0, 3)
    open_offset = rng.choice([-1, 3])
    open_col = start + open_offset
    if not (0 <= open_col < COLS):
        open_col = start + (3 if open_offset < 0 else -1)
    for i in range(3):
        board.grid[ROWS - 1][start + i] = "Y"
    # Ensure open_col is empty and legal (bottom open)
    board.grid[ROWS - 1][open_col] = "."
    # Fill other random noise pieces that don't complete wins
    for _ in range(rng.randint(0, 4)):
        c = rng.randint(0, COLS - 1)
        if c == open_col:
            continue
        board.drop_piece(c, rng.choice(["R", "Y"]))
    threat = board.find_immediate_threat("Y")
    if threat is None:
        # Force clean bottom threat
        board = Connect4Board([["." for _ in range(COLS)] for _ in range(ROWS)], turn="R")
        for i in range(3):
            board.grid[ROWS - 1][start + i] = "Y"
        threat = board.find_immediate_threat("Y")
        assert threat is not None
    return board, "R", threat


def generate_connect4_samples(
    n_samples: int,
    *,
    seed: int = 42,
    hard: bool = False,
) -> list[DistillSample]:
    """Connect Four distill with diversified drops + balanced threat alerts."""
    rng = random.Random(seed)
    samples: list[DistillSample] = []
    alert_yes = 0
    alert_no = 0
    yes_budget = max(1, n_samples // 5)
    drop_counts: dict[str, int] = {str(c + 1): 0 for c in range(COLS)}

    def _maybe_add_drop(
        *,
        state: dict[str, Any],
        label_1based: str,
        to_move: str,
        mirror_h: bool,
        forced_block: bool = False,
    ) -> None:
        nonlocal samples
        if len(samples) >= n_samples:
            return
        # Soft balance: skip free-drop samples that widen the majority gap too far
        if not forced_block:
            vals = list(drop_counts.values())
            majority = max(vals) if vals else 0
            minority = min(vals) if vals else 0
            this = drop_counts.get(label_1based, 0)
            if this > minority + max(40, n_samples // 80) and this >= majority:
                return
        options = {str(c + 1): f"Drop in Column {c + 1}" for c in range(COLS)}
        s = choice_sample(
            task="connect4.drop",
            state=state,
            instructions=(
                "Inspect the 7x6 Connect Four board. "
                "Choose the column (1 to 7) to drop your disc."
            ),
            options=options,
            label_key=label_1based,
            meta={
                "gym": "connect4",
                "turn": to_move,
                "aug_mirror_h": mirror_h,
                **({"forced_block": True} if forced_block else {}),
            },
            rng=rng,
            hard=hard,
        )
        samples.append(s)
        drop_counts[label_1based] = drop_counts.get(label_1based, 0) + 1

    while len(samples) < n_samples:
        # Top-up explicit threat-yes (+ matching block drop)
        if alert_yes < yes_budget and alert_yes <= alert_no and rng.random() < 0.4:
            board, to_move, block_col = make_horizontal_threat_board(rng)
            from systemone_lite.synth.augmentation import apply_connect4_mirror

            mirror_h = rng.random() < 0.5
            if mirror_h:
                aug_grid, aug_block = apply_connect4_mirror(board.grid, block_col + 1)
                aug_board = Connect4Board(grid=aug_grid)
            else:
                aug_board = board
                aug_block = block_col + 1
            state = {
                "grid_map": f"\n{aug_board.render_ascii()}\n",
                "turn": "Red (R)" if to_move == "R" else "Yellow (Y)",
                "legend": "'R': Red disc, 'Y': Yellow disc, '.': Empty slot. Columns are 1 to 7.",
            }
            samples.append(
                _threat_alert(
                    state,
                    True,
                    extra_meta={"threat_synth": True, "aug_mirror_h": mirror_h},
                )
            )
            alert_yes += 1
            _maybe_add_drop(
                state=state,
                label_1based=str(aug_block),
                to_move=to_move,
                mirror_h=mirror_h,
                forced_block=True,
            )
            continue

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
            best_col = best_move_connect4(
                board,
                curr_player,
                rng=rng,
                column_bias=_drop_column_weights(drop_counts),
            )

            from systemone_lite.synth.augmentation import apply_connect4_mirror

            mirror_h = rng.random() < 0.5
            if mirror_h:
                aug_grid, aug_best_col_1based = apply_connect4_mirror(board.grid, best_col + 1)
                aug_board = Connect4Board(grid=aug_grid)
            else:
                aug_board = board
                aug_best_col_1based = best_col + 1

            state = {
                "grid_map": f"\n{aug_board.render_ascii()}\n",
                "turn": f"{'Red (R)' if curr_player == 'R' else 'Yellow (Y)'}",
                "legend": "'R': Red disc, 'Y': Yellow disc, '.': Empty slot. Columns are 1 to 7.",
            }

            if rng.random() < 0.70:
                _maybe_add_drop(
                    state=state,
                    label_1based=str(aug_best_col_1based),
                    to_move=curr_player,
                    mirror_h=mirror_h,
                    forced_block=False,
                )

            threat_col = aug_board.find_immediate_threat(opp_player)
            has_threat = threat_col is not None
            if has_threat and alert_yes <= alert_no and len(samples) < n_samples:
                samples.append(_threat_alert(state, True))
                alert_yes += 1
            elif (
                (not has_threat)
                and alert_no <= alert_yes
                and len(samples) < n_samples
                and rng.random() < 0.35
            ):
                samples.append(_threat_alert(state, False))
                alert_no += 1

            r = board.drop_piece(best_col, curr_player)
            if r is not None and board.check_win_at(r, best_col, curr_player):
                break

    return samples[:n_samples]
