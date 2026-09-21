"""Conway-style cellular automata distill (verifiable multi-step grid dynamics).

System One choice probes — not full-grid generation. Horizon ``n`` is the
difficulty knob; prefer short ``n`` with occasional harder draws.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from systemone_lite.chess_data import DistillSample
from systemone_lite.synth.common import choice_sample

# Named Life-like rules: birth / survive neighbor counts.
RULE_PRESETS: dict[str, tuple[frozenset[int], frozenset[int]]] = {
    "B3/S23": (frozenset({3}), frozenset({2, 3})),  # Conway
    "B36/S23": (frozenset({3, 6}), frozenset({2, 3})),  # HighLife-ish birth
    "B3/S123": (frozenset({3}), frozenset({1, 2, 3})),
}


@dataclass
class CAGrid:
    cells: list[list[int]]  # 0/1
    birth: frozenset[int]
    survive: frozenset[int]
    rule_name: str

    @property
    def height(self) -> int:
        return len(self.cells)

    @property
    def width(self) -> int:
        return len(self.cells[0]) if self.cells else 0

    def copy(self) -> CAGrid:
        return CAGrid(
            cells=[list(r) for r in self.cells],
            birth=self.birth,
            survive=self.survive,
            rule_name=self.rule_name,
        )

    def live_count(self) -> int:
        return sum(sum(r) for r in self.cells)

    def neighbor_count(self, r: int, c: int) -> int:
        h, w = self.height, self.width
        total = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    total += self.cells[rr][cc]
        return total

    def step(self) -> CAGrid:
        nxt = [[0] * self.width for _ in range(self.height)]
        for r in range(self.height):
            for c in range(self.width):
                n = self.neighbor_count(r, c)
                if self.cells[r][c]:
                    nxt[r][c] = 1 if n in self.survive else 0
                else:
                    nxt[r][c] = 1 if n in self.birth else 0
        return CAGrid(nxt, self.birth, self.survive, self.rule_name)

    def after(self, n: int) -> CAGrid:
        g = self
        for _ in range(n):
            g = g.step()
        return g

    def render_ascii(self, *, alive: str = "O", dead: str = ".") -> str:
        return "\n".join("".join(alive if c else dead for c in row) for row in self.cells)


def _random_grid(
    rng: random.Random,
    *,
    width: int,
    height: int,
    density: float,
    rule_name: str,
) -> CAGrid:
    birth, survive = RULE_PRESETS[rule_name]
    cells = [
        [1 if rng.random() < density else 0 for _ in range(width)]
        for _ in range(height)
    ]
    # Avoid all-dead / all-alive trivial boards
    live = sum(sum(r) for r in cells)
    if live == 0:
        cells[rng.randrange(height)][rng.randrange(width)] = 1
    if live == width * height:
        cells[rng.randrange(height)][rng.randrange(width)] = 0
    return CAGrid(cells, birth, survive, rule_name)


def _rule_legend(rule_name: str) -> str:
    birth, survive = RULE_PRESETS[rule_name]
    b = "".join(str(x) for x in sorted(birth))
    s = "".join(str(x) for x in sorted(survive))
    return (
        f"Cellular automaton rule {rule_name} (B{b}/S{s}): "
        f"a dead cell becomes live with exactly {{{', '.join(map(str, sorted(birth)))}}} neighbors; "
        f"a live cell stays live with {{{', '.join(map(str, sorted(survive)))}}} neighbors; "
        f"otherwise dies/stays dead. 'O'=live, '.'=dead. Torus=no (edges have fewer neighbors)."
    )


def _state(grid: CAGrid, n_steps: int) -> dict[str, Any]:
    return {
        "grid_map": f"\n{grid.render_ascii()}\n",
        "n_steps": n_steps,
        "rule": grid.rule_name,
        "legend": _rule_legend(grid.rule_name),
    }


def _sample_n(rng: random.Random) -> int:
    # Curriculum: mostly short horizon
    roll = rng.random()
    if roll < 0.45:
        return 1
    if roll < 0.75:
        return rng.randint(2, 3)
    if roll < 0.92:
        return rng.randint(4, 6)
    return rng.randint(7, 8)


def generate_ca_samples(
    n_samples: int,
    *,
    seed: int = 42,
    hard: bool = False,
    keep_conway_prob: float = 0.7,
) -> list[DistillSample]:
    """Generate CA choice samples (cell fate, population parity, pop change)."""
    rng = random.Random(seed)
    samples: list[DistillSample] = []
    cell_yes = cell_no = 0
    pop_up = pop_down = pop_same = 0
    task_rr = ["ca.cell_alive", "ca.pop_parity", "ca.pop_change"]
    rr_i = 0

    while len(samples) < n_samples:
        w = rng.randint(6, 10)
        h = rng.randint(6, 10)
        density = rng.uniform(0.28, 0.48)
        if rng.random() < keep_conway_prob:
            rule = "B3/S23"
        else:
            rule = rng.choice([k for k in RULE_PRESETS if k != "B3/S23"])
        grid0 = _random_grid(rng, width=w, height=h, density=density, rule_name=rule)
        n = _sample_n(rng)
        grid_n = grid0.after(n)
        state = _state(grid0, n)
        meta_base = {
            "gym": "cellular_automata",
            "rule": rule,
            "n_steps": n,
            "grid_size": f"{h}x{w}",
        }

        task = task_rr[rr_i % 3]
        rr_i += 1

        if task == "ca.cell_alive":
            live_cells = [
                (r, c)
                for r in range(h)
                for c in range(w)
                if grid_n.cells[r][c]
            ]
            dead_cells = [
                (r, c)
                for r in range(h)
                for c in range(w)
                if not grid_n.cells[r][c]
            ]
            want_yes = cell_yes <= cell_no
            if want_yes and live_cells:
                r, c = rng.choice(live_cells)
            elif (not want_yes) and dead_cells:
                r, c = rng.choice(dead_cells)
            else:
                r, c = rng.randrange(h), rng.randrange(w)
            alive = bool(grid_n.cells[r][c])
            label = "yes" if alive else "no"
            st = dict(state)
            st["query_cell"] = {"row": r, "col": c, "index_origin": "0-based top-left"}
            s = choice_sample(
                task=task,
                state=st,
                instructions=(
                    f"Apply the legend rule for exactly {n} step(s). "
                    f"Will the cell at row={r}, col={c} (0-based) be live?"
                ),
                options={"yes": "Cell will be live (O)", "no": "Cell will be dead (.)"},
                label_key=label,
                meta={**meta_base, "query_cell": [r, c]},
                rng=rng,
                hard=hard,
            )
            if alive:
                cell_yes += 1
            else:
                cell_no += 1
        elif task == "ca.pop_parity":
            parity = "even" if grid_n.live_count() % 2 == 0 else "odd"
            s = choice_sample(
                task=task,
                state=state,
                instructions=(
                    f"Apply the legend rule for exactly {n} step(s). "
                    "Will the number of live cells be even or odd?"
                ),
                options={"even": "Even live count", "odd": "Odd live count"},
                label_key=parity,
                meta={**meta_base, "live_after": grid_n.live_count()},
                rng=rng,
                hard=hard,
            )
        else:
            before = grid0.live_count()
            after = grid_n.live_count()
            if after > before:
                label = "up"
            elif after < before:
                label = "down"
            else:
                label = "same"
            counts = {"up": pop_up, "down": pop_down, "same": pop_same}
            if label != "same" and counts[label] > min(pop_up, pop_down) + 8:
                continue
            if label == "same" and pop_same > min(pop_up, pop_down) // 2 + 5:
                continue
            s = choice_sample(
                task=task,
                state=state,
                instructions=(
                    f"Apply the legend rule for exactly {n} step(s). "
                    "Compared to the initial live count, does the population go up, down, or stay the same?"
                ),
                options={
                    "up": "More live cells",
                    "down": "Fewer live cells",
                    "same": "Same live count",
                },
                label_key=label,
                meta={**meta_base, "live_before": before, "live_after": after},
                rng=rng,
                hard=hard,
            )
            if label == "up":
                pop_up += 1
            elif label == "down":
                pop_down += 1
            else:
                pop_same += 1

        samples.append(s)

    return samples
