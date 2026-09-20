"""Sokoban (box-pushing puzzle) 2D text map generator & solver for System One."""

from __future__ import annotations

import collections
import random
from dataclasses import dataclass
from typing import Any

from systemone_lite.chess_data import DistillSample, alias_criteria
from systemone_lite.synth.common import choice_sample

DIRECTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


@dataclass
class SokobanLevel:
    grid: list[list[str]]  # '#' wall, ' ' floor
    player: tuple[int, int]
    boxes: set[tuple[int, int]]
    targets: set[tuple[int, int]]

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0])

    def render_ascii(self) -> str:
        lines = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                pos = (r, c)
                is_target = pos in self.targets
                is_box = pos in self.boxes
                is_player = pos == self.player

                if self.grid[r][c] == "#":
                    row.append("#")
                elif is_box and is_target:
                    row.append("*")  # Box on target
                elif is_player and is_target:
                    row.append("+")  # Player on target
                elif is_box:
                    row.append("$")  # Box
                elif is_target:
                    row.append(".")  # Target
                elif is_player:
                    row.append("@")  # Player
                else:
                    row.append(" ")  # Empty floor
            lines.append("".join(row))
        return "\n".join(lines)

    def is_solved(self) -> bool:
        return self.boxes == self.targets

    def is_corner_deadlock(self, box: tuple[int, int]) -> bool:
        """A box in a non-target corner is permanently stuck."""
        if box in self.targets:
            return False
        r, c = box
        # Check 4 corners
        up_wall = self.grid[r - 1][c] == "#"
        down_wall = self.grid[r + 1][c] == "#"
        left_wall = self.grid[r][c - 1] == "#"
        right_wall = self.grid[r][c + 1] == "#"

        if (up_wall and left_wall) or (up_wall and right_wall) or \
           (down_wall and left_wall) or (down_wall and right_wall):
            return True
        return False

    def has_any_deadlock(self) -> bool:
        return any(self.is_corner_deadlock(b) for b in self.boxes)

    def get_legal_actions(self) -> dict[str, tuple[tuple[int, int], set[tuple[int, int]], bool]]:
        """Returns dict of action -> (next_player, next_boxes, pushed_box)."""
        actions = {}
        pr, pc = self.player
        for dname, (dr, dc) in DIRECTIONS.items():
            nr, nc = pr + dr, pc + dc
            if not (0 <= nr < self.height and 0 <= nc < self.width):
                continue
            if self.grid[nr][nc] == "#":
                continue

            pos = (nr, nc)
            if pos in self.boxes:
                # Pushing box
                nnr, nnc = nr + dr, nc + dc
                if not (0 <= nnr < self.height and 0 <= nnc < self.width):
                    continue
                if self.grid[nnr][nnc] == "#" or (nnr, nnc) in self.boxes:
                    continue  # Blocked
                new_boxes = set(self.boxes)
                new_boxes.remove(pos)
                new_boxes.add((nnr, nnc))
                actions[dname] = ((nr, nc), new_boxes, True)
            else:
                # Regular walk
                actions[dname] = ((nr, nc), set(self.boxes), False)
        return actions


# Curated micro Sokoban puzzle templates (playable & solvable)
MICRO_LEVELS = [
    [
        "######",
        "# .  #",
        "# $  #",
        "#  @ #",
        "######",
    ],
    [
        "#######",
        "# . $ #",
        "#   @ #",
        "#  $  #",
        "#  .  #",
        "#######",
    ],
    [
        "#######",
        "#  .  #",
        "# $#$ #",
        "#  @  #",
        "#  .  #",
        "#######",
    ],
    [
        "######",
        "#@ $ #",
        "#  $ #",
        "# .. #",
        "######",
    ],
    [
        "#######",
        "#  .  #",
        "# $ $ #",
        "# .@  #",
        "#######",
    ],
    [
        "######",
        "#  . #",
        "# @$ #",
        "#    #",
        "######",
    ],
    [
        "#######",
        "# @ $ #",
        "# #.  #",
        "#     #",
        "#######",
    ],
    [
        "########",
        "#  . $ #",
        "#  #$  #",
        "#  @.  #",
        "########",
    ],
    [
        "#######",
        "# .   #",
        "#  $  #",
        "#  #  #",
        "# @   #",
        "#######",
    ],
    [
        "#######",
        "# @   #",
        "#  $  #",
        "#  .  #",
        "#     #",
        "#######",
    ],
    [
        "########",
        "# .  $ #",
        "#  @   #",
        "#  . $ #",
        "########",
    ],
    [
        "#######",
        "#  .  #",
        "#  $  #",
        "# @$  #",
        "#  .  #",
        "#######",
    ],
    [
        "######",
        "# .  #",
        "# #$ #",
        "# @  #",
        "######",
    ],
    [
        "########",
        "#   .  #",
        "# @ $  #",
        "#  $ . #",
        "########",
    ],
    [
        "#######",
        "#  .  #",
        "#  @$ #",
        "#  .  #",
        "#######",
    ],
    [
        "########",
        "#  .   #",
        "#  $   #",
        "# @# $ #",
        "#    . #",
        "########",
    ],
    [
        "#######",
        "#  .  #",
        "#  $  #",
        "#  .  #",
        "#  $@ #",
        "#######",
    ],
    [
        "######",
        "#  . #",
        "#  $ #",
        "# @. #",
        "######",
    ],
    [
        "#######",
        "# @ $ #",
        "#  .  #",
        "# $ . #",
        "#######",
    ],
    [
        "########",
        "# .  @ #",
        "#  $   #",
        "#   $  #",
        "#   .  #",
        "########",
    ],
]


def parse_level(lines: list[str]) -> SokobanLevel:
    grid = []
    player = (0, 0)
    boxes = set()
    targets = set()

    for r, line in enumerate(lines):
        row = []
        for c, ch in enumerate(line):
            if ch == "#":
                row.append("#")
            else:
                row.append(" ")
                if ch == "@":
                    player = (r, c)
                elif ch == "$":
                    boxes.add((r, c))
                elif ch == ".":
                    targets.add((r, c))
                elif ch == "*":
                    boxes.add((r, c))
                    targets.add((r, c))
                elif ch == "+":
                    player = (r, c)
                    targets.add((r, c))
        grid.append(row)
    return SokobanLevel(grid=grid, player=player, boxes=boxes, targets=targets)


def solve_sokoban_bfs(level: SokobanLevel, max_nodes: int = 1500) -> list[str] | None:
    """BFS shortest path solver for small Sokoban levels."""
    start_state = (level.player, tuple(sorted(level.boxes)))
    queue = collections.deque([(start_state, [])])
    visited = {start_state}

    target_tuple = tuple(sorted(level.targets))

    while queue and len(visited) < max_nodes:
        (curr_player, curr_boxes_tuple), path = queue.popleft()
        curr_boxes = set(curr_boxes_tuple)

        if curr_boxes_tuple == target_tuple:
            return path

        curr_lvl = SokobanLevel(
            grid=level.grid,
            player=curr_player,
            boxes=curr_boxes,
            targets=level.targets,
        )

        for dname, (next_player, next_boxes, pushed) in curr_lvl.get_legal_actions().items():
            # Prune deadlocks immediately
            if pushed:
                temp_lvl = SokobanLevel(level.grid, next_player, next_boxes, level.targets)
                if temp_lvl.has_any_deadlock():
                    continue

            next_state = (next_player, tuple(sorted(next_boxes)))
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [dname]))
    return None


def generate_sokoban_samples(
    n_samples: int,
    *,
    seed: int = 42,
    hard: bool = False,
) -> list[DistillSample]:
    """Generate synthetic supervised choice/noul/score samples for Sokoban."""
    rng = random.Random(seed)
    samples: list[DistillSample] = []

    while len(samples) < n_samples:
        template = rng.choice(MICRO_LEVELS)
        level = parse_level(template)

        # Solve level to get optimal action sequence
        path = solve_sokoban_bfs(level)
        if not path:
            continue

        curr_level = SokobanLevel(
            grid=[list(r) for r in level.grid],
            player=level.player,
            boxes=set(level.boxes),
            targets=set(level.targets),
        )

        for best_action in path:
            if len(samples) >= n_samples:
                break

            grid_map = f"\n{curr_level.render_ascii()}\n"
            legal_actions = curr_level.get_legal_actions()
            if not legal_actions or best_action not in legal_actions:
                break

            state = {
                "grid_map": grid_map,
                "legend": "'#': Wall, '@': Player, '$': Box, '.': Target, '*': Box on target",
                "player_position": f"Row {curr_level.player[0]}, Col {curr_level.player[1]}",
                "boxes_placed": f"{len(curr_level.boxes & curr_level.targets)} of {len(curr_level.targets)}",
                "remaining_targets": len(curr_level.targets - curr_level.boxes),
            }

            # 1. Choice sample: which direction to move/push
            action_options = {}
            for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
                if d in legal_actions:
                    _, _, pushed = legal_actions[d]
                    action_options[d] = f"Push box {d}" if pushed else f"Walk {d}"
                else:
                    action_options[d] = f"Blocked wall or obstacle {d}"

            # Randomly add choice sample
            if rng.random() < 0.70:
                s = choice_sample(
                    task="sokoban.direction",
                    state=state,
                    instructions=(
                        "Inspect the 2D Sokoban map. Choose the best move direction "
                        "for player (@) to solve the puzzle without causing a deadlock."
                    ),
                    options=action_options,
                    label_key=best_action,
                    meta={"gym": "sokoban", "level_size": f"{curr_level.height}x{curr_level.width}"},
                    rng=rng,
                    hard=hard,
                )
                samples.append(s)

            # 2. Noul sample: deadlock check
            if len(samples) < n_samples and rng.random() < 0.40:
                # Check if any illegal or dangerous push causes deadlock
                has_deadlock = curr_level.has_any_deadlock()
                criteria, alias_map = alias_criteria({"yes": "Deadlock detected", "no": "State is safe"})
                key_to_alias = {v: k for k, v in alias_map.items()}
                lbl_key = "yes" if has_deadlock else "no"
                samples.append(
                    DistillSample(
                        task="sokoban.deadlock_alert",
                        state=state,
                        instructions="Is any box currently trapped in an irreversible corner deadlock?",
                        criteria=criteria,
                        label_alias=key_to_alias[lbl_key],
                        label_key=lbl_key,
                        meta={"gym": "sokoban", "schema": "noul"},
                    )
                )

            # Advance state along path
            next_player, next_boxes, _ = legal_actions[best_action]
            curr_level = SokobanLevel(
                grid=curr_level.grid,
                player=next_player,
                boxes=next_boxes,
                targets=curr_level.targets,
            )

    return samples[:n_samples]
