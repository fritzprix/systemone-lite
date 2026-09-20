"""GridWorld Hazards (Lava/Spike maze navigation) 2D text map generator & solver."""

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
class GridWorldMap:
    grid: list[list[str]]  # '#' wall, '.' floor, 'X' hazard, 'G' goal, '@' player
    player: tuple[int, int]
    goal: tuple[int, int]
    hazards: set[tuple[int, int]]

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
                if pos == self.player:
                    row.append("@")
                elif pos == self.goal:
                    row.append("G")
                elif pos in self.hazards:
                    row.append("X")
                else:
                    row.append(self.grid[r][c])
            lines.append(" ".join(row))
        return "\n".join(lines)

    def is_adjacent_to_hazard(self) -> bool:
        pr, pc = self.player
        for dr, dc in DIRECTIONS.values():
            if (pr + dr, pc + dc) in self.hazards:
                return True
        return False

    def find_shortest_safe_path(self) -> list[str] | None:
        """BFS shortest safe path from player to goal avoiding walls and hazards."""
        queue = collections.deque([(self.player, [])])
        visited = {self.player}

        while queue:
            (cr, cc), path = queue.popleft()
            if (cr, cc) == self.goal:
                return path

            for dname, (dr, dc) in DIRECTIONS.items():
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < self.height and 0 <= nc < self.width):
                    continue
                if self.grid[nr][nc] == "#" or (nr, nc) in self.hazards:
                    continue
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [dname]))
        return None


def generate_random_gridworld(
    width: int = 8,
    height: int = 8,
    hazard_count: int = 4,
    rng: random.Random | None = None,
) -> GridWorldMap:
    if rng is None:
        rng = random.Random()

    while True:
        grid = [["." for _ in range(width)] for _ in range(height)]
        # Outer boundary walls
        for r in range(height):
            grid[r][0] = "#"
            grid[r][width - 1] = "#"
        for c in range(width):
            grid[0][c] = "#"
            grid[height - 1][c] = "#"

        # Internal obstacle walls
        for _ in range(rng.randint(3, 7)):
            wr = rng.randint(1, height - 2)
            wc = rng.randint(1, width - 2)
            grid[wr][wc] = "#"

        player = (1, 1)
        goal = (height - 2, width - 2)
        grid[player[0]][player[1]] = "."
        grid[goal[0]][goal[1]] = "."

        empty_cells = [
            (r, c)
            for r in range(1, height - 1)
            for c in range(1, width - 1)
            if (r, c) != player and (r, c) != goal and grid[r][c] == "."
        ]

        if len(empty_cells) < hazard_count:
            continue

        hazards = set(rng.sample(empty_cells, hazard_count))
        m = GridWorldMap(grid=grid, player=player, goal=goal, hazards=hazards)
        path = m.find_shortest_safe_path()
        if path:
            return m


def generate_gridworld_samples(
    n_samples: int,
    *,
    seed: int = 42,
    hard: bool = False,
) -> list[DistillSample]:
    """Generate synthetic supervised choice/noul/score samples for GridWorld."""
    rng = random.Random(seed)
    samples: list[DistillSample] = []

    while len(samples) < n_samples:
        w = rng.randint(7, 9)
        h = rng.randint(7, 9)
        num_hazards = rng.randint(3, 6)
        m = generate_random_gridworld(width=w, height=h, hazard_count=num_hazards, rng=rng)

        path = m.find_shortest_safe_path()
        if not path:
            continue

        curr_map = GridWorldMap(
            grid=[list(r) for r in m.grid],
            player=m.player,
            goal=m.goal,
            hazards=set(m.hazards),
        )

        for best_action in path:
            if len(samples) >= n_samples:
                break

            # Apply D4 augmentation to rendered grid and action
            from systemone_lite.synth.augmentation import apply_d4_transform
            rot_k = rng.randint(0, 3)
            flip_h = rng.random() < 0.5
            raw_lines = [list(r) for r in curr_map.render_ascii().split("\n") if r]
            aug_matrix, aug_best_action = apply_d4_transform(raw_lines, best_action, rot_k=rot_k, flip_h=flip_h)
            aug_grid_map = "\n" + "\n".join("".join(r) for r in aug_matrix) + "\n"

            options = {d: f"Move {d}" for d in DIRECTIONS}

            state = {
                "grid_map": aug_grid_map,
                "legend": "'#': Wall, '@': Player, 'G': Goal, 'X': Deadly Trap, '.': Floor",
            }

            # 1. Choice sample: strictly bare directional options
            if rng.random() < 0.70:
                s = choice_sample(
                    task="gridworld.move",
                    state=state,
                    instructions="Inspect the 2D GridWorld map. Choose the move direction: UP, DOWN, LEFT, RIGHT.",
                    options=options,
                    label_key=aug_best_action,
                    meta={
                        "gym": "gridworld",
                        "grid_size": f"{h}x{w}",
                        "aug_rot_k": rot_k,
                        "aug_flip_h": flip_h,
                    },
                    rng=rng,
                    hard=hard,
                )
                samples.append(s)

            # 2. Noul sample: hazard proximity alert
            if len(samples) < n_samples and rng.random() < 0.40:
                near_hazard = curr_map.is_adjacent_to_hazard()
                criteria, alias_map = alias_criteria({"yes": "Immediate deadly trap nearby", "no": "Surroundings safe"})
                key_to_alias = {v: k for k, v in alias_map.items()}
                lbl_key = "yes" if near_hazard else "no"
                samples.append(
                    DistillSample(
                        task="gridworld.hazard_alert",
                        state=state,
                        instructions="Is the player (@) currently adjacent to a deadly trap (X)?",
                        criteria=criteria,
                        label_alias=key_to_alias[lbl_key],
                        label_key=lbl_key,
                        meta={"gym": "gridworld", "schema": "noul"},
                    )
                )

            # Advance along path
            pr, pc = curr_map.player
            dr, dc = DIRECTIONS[best_action]
            curr_map.player = (pr + dr, pc + dc)

    return samples[:n_samples]
