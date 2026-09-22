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
            if pushed:
                temp_lvl = SokobanLevel(level.grid, next_player, next_boxes, level.targets)
                if temp_lvl.has_any_deadlock():
                    continue

            next_state = (next_player, tuple(sorted(next_boxes)))
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [dname]))
    return None


def _floor_cells(grid: list[list[str]]) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r, row in enumerate(grid)
        for c, ch in enumerate(row)
        if ch != "#"
    ]


def _corner_cells(grid: list[list[str]]) -> list[tuple[int, int]]:
    """Interior floor cells with two orthogonal walls (classic Sokoban corner)."""
    corners: list[tuple[int, int]] = []
    h, w = len(grid), len(grid[0])
    for r, c in _floor_cells(grid):
        up = r - 1 < 0 or grid[r - 1][c] == "#"
        down = r + 1 >= h or grid[r + 1][c] == "#"
        left = c - 1 < 0 or grid[r][c - 1] == "#"
        right = c + 1 >= w or grid[r][c + 1] == "#"
        if (up and left) or (up and right) or (down and left) or (down and right):
            corners.append((r, c))
    return corners


def generate_procedural_level(rng: random.Random) -> SokobanLevel | None:
    """Small random room with 1–2 boxes; must be BFS-solvable."""
    for _ in range(80):
        w = rng.randint(6, 8)
        h = rng.randint(5, 7)
        grid = [[" " for _ in range(w)] for _ in range(h)]
        for r in range(h):
            grid[r][0] = "#"
            grid[r][w - 1] = "#"
        for c in range(w):
            grid[0][c] = "#"
            grid[h - 1][c] = "#"
        for _ in range(rng.randint(0, 3)):
            rr = rng.randint(1, h - 2)
            cc = rng.randint(1, w - 2)
            grid[rr][cc] = "#"

        floor = _floor_cells(grid)
        n_boxes = rng.choice([1, 1, 2])
        if len(floor) < n_boxes * 2 + 1:
            continue
        rng.shuffle(floor)
        boxes = set(floor[:n_boxes])
        targets = set(floor[n_boxes : n_boxes * 2])
        remain = [p for p in floor if p not in boxes and p not in targets]
        if not remain:
            continue
        # Prefer non-corner starts for boxes (avoid instant deadlock)
        corners = set(_corner_cells(grid))
        if any(b in corners and b not in targets for b in boxes):
            continue
        player = remain[0]
        level = SokobanLevel(grid=grid, player=player, boxes=boxes, targets=targets)
        path = solve_sokoban_bfs(level, max_nodes=2000)
        if path and 2 <= len(path) <= 24:
            return level
    return None


def make_corner_deadlock_level(
    template: SokobanLevel, rng: random.Random
) -> SokobanLevel | None:
    """Force at least one box into a non-target corner (positive deadlock label)."""
    corners = [c for c in _corner_cells(template.grid) if c not in template.targets]
    if not corners:
        return None
    dead_box = rng.choice(corners)
    other_boxes = [b for b in template.boxes if b != dead_box]
    # Keep other boxes on floor if possible
    floor = [p for p in _floor_cells(template.grid) if p != dead_box]
    boxes = {dead_box}
    for b in other_boxes:
        if not floor:
            break
        pick = rng.choice(floor)
        floor = [p for p in floor if p != pick]
        boxes.add(pick)
    remain = [p for p in _floor_cells(template.grid) if p not in boxes]
    if not remain:
        return None
    player = rng.choice(remain)
    lvl = SokobanLevel(
        grid=[list(r) for r in template.grid],
        player=player,
        boxes=boxes,
        targets=set(template.targets),
    )
    if not lvl.has_any_deadlock():
        return None
    return lvl


def _render_aug_state(
    level: SokobanLevel, rng: random.Random
) -> tuple[dict[str, Any], int, bool, dict[str, str]]:
    from systemone_lite.synth.augmentation import apply_d4_transform
    from systemone_lite.synth.symbol_aug import maybe_remap_sokoban_map

    rot_k = rng.randint(0, 3)
    flip_h = rng.random() < 0.5
    raw_lines = [list(r) for r in level.render_ascii().split("\n") if r]
    aug_matrix, _ = apply_d4_transform(raw_lines, None, rot_k=rot_k, flip_h=flip_h)
    aug_matrix, legend, role_syms = maybe_remap_sokoban_map(rng, aug_matrix)
    state = {
        "grid_map": "\n" + "\n".join("".join(r) for r in aug_matrix) + "\n",
        "legend": legend,
    }
    return state, rot_k, flip_h, role_syms


def _deadlock_alert(
    state: dict[str, Any],
    has_deadlock: bool,
    *,
    extra_meta: dict[str, Any] | None = None,
) -> DistillSample:
    criteria, alias_map = alias_criteria(
        {"yes": "Deadlock detected", "no": "State is safe"}
    )
    key_to_alias = {v: k for k, v in alias_map.items()}
    lbl_key = "yes" if has_deadlock else "no"
    meta: dict[str, Any] = {"gym": "sokoban", "schema": "noul"}
    if extra_meta:
        meta.update(extra_meta)
    return DistillSample(
        task="sokoban.deadlock_alert",
        state=state,
        instructions="Is any box currently trapped in an irreversible corner deadlock?",
        criteria=criteria,
        label_alias=key_to_alias[lbl_key],
        label_key=lbl_key,
        meta=meta,
    )


def _pick_level(rng: random.Random) -> SokobanLevel | None:
    """50% curated micro templates, 50% procedural rooms."""
    if rng.random() < 0.5:
        return parse_level(rng.choice(MICRO_LEVELS))
    return generate_procedural_level(rng)


def generate_sokoban_samples(
    n_samples: int,
    *,
    seed: int = 42,
    hard: bool = False,
) -> list[DistillSample]:
    """Sokoban distill with solvable BFS moves + **balanced** deadlock alerts.

    Deadlock `yes` examples are synthesized by parking a box in a non-target
    corner (previously impossible: alerts were only taken on optimal paths).
    """
    rng = random.Random(seed)
    samples: list[DistillSample] = []
    alert_yes = 0
    alert_no = 0
    # Reserve ~30% of budget for explicit deadlock-yes boards
    yes_budget = max(1, n_samples // 5)

    while len(samples) < n_samples:
        # Top-up positive deadlock class first if behind
        if alert_yes < yes_budget and alert_yes <= alert_no and rng.random() < 0.45:
            base = _pick_level(rng)
            if base is None:
                base = parse_level(rng.choice(MICRO_LEVELS))
            dead = make_corner_deadlock_level(base, rng)
            if dead is not None:
                state, rot_k, flip_h, role_syms = _render_aug_state(dead, rng)
                samples.append(
                    _deadlock_alert(
                        state,
                        True,
                        extra_meta={
                            "aug_rot_k": rot_k,
                            "aug_flip_h": flip_h,
                            "deadlock_synth": True,
                            "symbol_remap": role_syms,
                        },
                    )
                )
                alert_yes += 1
                continue

        level = _pick_level(rng)
        if level is None:
            level = parse_level(rng.choice(MICRO_LEVELS))

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

            from systemone_lite.synth.augmentation import apply_d4_transform
            from systemone_lite.synth.symbol_aug import maybe_remap_sokoban_map

            rot_k = rng.randint(0, 3)
            flip_h = rng.random() < 0.5
            raw_lines = [list(r) for r in curr_level.render_ascii().split("\n") if r]
            aug_matrix, aug_best_action = apply_d4_transform(
                raw_lines, best_action, rot_k=rot_k, flip_h=flip_h
            )
            aug_matrix, legend, role_syms = maybe_remap_sokoban_map(rng, aug_matrix)
            aug_grid_map = "\n" + "\n".join("".join(r) for r in aug_matrix) + "\n"
            state = {
                "grid_map": aug_grid_map,
                "legend": legend,
            }
            action_options = {d: f"Move {d}" for d in ["UP", "DOWN", "LEFT", "RIGHT"]}

            if rng.random() < 0.70:
                s = choice_sample(
                    task="sokoban.direction",
                    state=state,
                    instructions=(
                        "Inspect the 2D Sokoban map. "
                        "Choose the move direction: UP, DOWN, LEFT, RIGHT."
                    ),
                    options=action_options,
                    label_key=aug_best_action or best_action,
                    meta={
                        "gym": "sokoban",
                        "level_size": f"{curr_level.height}x{curr_level.width}",
                        "aug_rot_k": rot_k,
                        "aug_flip_h": flip_h,
                        "symbol_remap": role_syms,
                    },
                    rng=rng,
                    hard=hard,
                )
                samples.append(s)

            # Safe-path alerts → "no"; keep roughly matched to yes count
            if (
                len(samples) < n_samples
                and not curr_level.has_any_deadlock()
                and alert_no <= alert_yes + 2
                and rng.random() < 0.35
            ):
                alert = _deadlock_alert(state, False)
                alert.meta["symbol_remap"] = role_syms
                samples.append(alert)
                alert_no += 1

            legal_actions = curr_level.get_legal_actions()
            if best_action not in legal_actions:
                break
            next_player, next_boxes, _ = legal_actions[best_action]
            curr_level = SokobanLevel(
                grid=curr_level.grid,
                player=next_player,
                boxes=next_boxes,
                targets=curr_level.targets,
            )

    return samples[:n_samples]
