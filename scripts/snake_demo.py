#!/usr/bin/env python3
"""Autonomous Snake game demo powered by systemone-lite.

System One acts as the real-time 'System 1' brain for the snake,
evaluating board state and selecting safe, food-seeking directions
in 10-25ms per step without autoregressive text generation.

Supports:
  - Rich ANSI animated terminal UI
  - Real-time decision statistics (probabilities, confidence, latency)
  - Multi-question evaluation (direction choice + danger score + food reachability noul)
  - In-process engine, HTTP server (--base-url), or deterministic stub (--stub)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from systemone_lite import SystemOneClient, choice, noul, score
from systemone_lite.stub import StubEngine

# Direction vectors: (row_delta, col_delta)
DIRECTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

OPPOSITE = {
    "UP": "DOWN",
    "DOWN": "UP",
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
}

HEAD_ICONS = {
    "UP": "▲",
    "DOWN": "▼",
    "LEFT": "◄",
    "RIGHT": "►",
}

# ANSI Colors
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_DIM = "\033[2m"
CLR_GREEN = "\033[32m"
CLR_BRIGHT_GREEN = "\033[92m"
CLR_RED = "\033[91m"
CLR_YELLOW = "\033[93m"
CLR_CYAN = "\033[96m"
CLR_MAGENTA = "\033[95m"
CLR_GRAY = "\033[90m"
CLR_BG_DARK = "\033[48;5;235m"


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


@dataclass
class DirectionAnalysis:
    direction: str
    target: tuple[int, int]
    is_wall: bool
    is_body: bool
    is_neck: bool
    distance_before: int
    distance_after: int
    is_closer: bool

    @property
    def is_safe(self) -> bool:
        return not (self.is_wall or self.is_body or self.is_neck)

    def summary(self) -> str:
        if self.is_neck:
            return "DEADLY: Immediate 180-degree reverse into neck."
        if self.is_wall:
            return "DEADLY: Wall collision at boundary."
        if self.is_body:
            return "DEADLY: Collision with own body segment."
        if self.is_closer:
            return f"SAFE: Moves closer to food (dist {self.distance_before} -> {self.distance_after})."
        return f"SAFE: Open path, moves further from food (dist {self.distance_before} -> {self.distance_after})."


class SnakeGame:
    def __init__(self, width: int = 10, height: int = 10, seed: int | None = None) -> None:
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        mid_r = self.height // 2
        mid_c = self.width // 2
        # Head at center, initial body extends to the left
        self.snake: list[tuple[int, int]] = [
            (mid_r, mid_c),
            (mid_r, mid_c - 1),
            (mid_r, mid_c - 2),
        ]
        self.direction = "RIGHT"
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.death_reason = ""
        self.food = self._spawn_food()

    def _spawn_food(self) -> tuple[int, int]:
        occupied = set(self.snake)
        empty = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in occupied
        ]
        if not empty:
            return (-1, -1)  # Board completely full (Victory!)
        return self.rng.choice(empty)

    @property
    def head(self) -> tuple[int, int]:
        return self.snake[0]

    def manhattan_distance(self, p1: tuple[int, int], p2: tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def analyze_all_directions(self) -> dict[str, DirectionAnalysis]:
        hr, hc = self.head
        curr_dist = self.manhattan_distance(self.head, self.food)
        neck = self.snake[1] if len(self.snake) > 1 else None
        body_set = set(self.snake[:-1])  # Tail will move unless food is eaten

        analyses: dict[str, DirectionAnalysis] = {}
        for dname, (dr, dc) in DIRECTIONS.items():
            tr, tc = hr + dr, hc + dc
            is_wall = not (0 <= tr < self.height and 0 <= tc < self.width)
            is_neck = (tr, tc) == neck
            is_body = (tr, tc) in body_set and not is_neck
            new_dist = self.manhattan_distance((tr, tc), self.food) if not is_wall else 999
            is_closer = new_dist < curr_dist

            analyses[dname] = DirectionAnalysis(
                direction=dname,
                target=(tr, tc),
                is_wall=is_wall,
                is_body=is_body,
                is_neck=is_neck,
                distance_before=curr_dist,
                distance_after=new_dist,
                is_closer=is_closer,
            )
        return analyses

    def step(self, chosen_direction: str) -> bool:
        if self.game_over:
            return False

        self.steps += 1
        dr, dc = DIRECTIONS.get(chosen_direction, DIRECTIONS[self.direction])
        self.direction = chosen_direction
        hr, hc = self.head
        new_head = (hr + dr, hc + dc)

        # Check wall collision
        if not (0 <= new_head[0] < self.height and 0 <= new_head[1] < self.width):
            self.game_over = True
            self.death_reason = f"Hit wall at ({new_head[0]}, {new_head[1]})"
            return False

        # Check self collision (excluding current tail if not growing)
        will_eat = new_head == self.food
        body_to_check = self.snake if will_eat else self.snake[:-1]
        if new_head in body_to_check:
            self.game_over = True
            self.death_reason = f"Collided with own body at ({new_head[0]}, {new_head[1]})"
            return False

        # Move snake
        self.snake.insert(0, new_head)
        if will_eat:
            self.score += 1
            self.food = self._spawn_food()
            if self.food == (-1, -1):
                self.game_over = True
                self.death_reason = "Board cleared! Victory!"
                return False
        else:
            self.snake.pop()

        return True

    def get_grid_ascii(self) -> str:
        """Render the 2D grid as a text matrix for the language model."""
        rows = []
        body_set = set(self.snake[1:])
        head = self.head
        for r in range(self.height):
            row_cells = []
            for c in range(self.width):
                pos = (r, c)
                if pos == head:
                    row_cells.append("@")
                elif pos in body_set:
                    row_cells.append("o")
                elif pos == self.food:
                    row_cells.append("★")
                else:
                    row_cells.append(".")
            rows.append(" ".join(row_cells))
        return "\n".join(rows)

    def build_systemone_payload(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
        """Construct System One state and questions with full 2D text grid and spatial relations."""
        analyses = self.analyze_all_directions()
        curr_dist = self.manhattan_distance(self.head, self.food)
        hr, hc = self.head
        fr, fc = self.food
        grid_ascii = self.get_grid_ascii()

        # Explicit relative spatial direction
        rel_vert = "UP (North)" if fr < hr else ("DOWN (South)" if fr > hr else "level horizontally")
        rel_horiz = "LEFT (West)" if fc < hc else ("RIGHT (East)" if fc > hc else "level vertically")
        closer_moves = [d for d, a in analyses.items() if a.is_closer and a.is_safe]
        safe_moves = [d for d, a in analyses.items() if a.is_safe]

        # Standard directional options without heuristic sorting or hints
        all_dirs = ["UP", "DOWN", "LEFT", "RIGHT"]
        alias_to_dir = {d: d for d in all_dirs}
        criteria = {d: f"Move {d}" for d in all_dirs}

        state = f"""Snake Grid Map:
{grid_ascii}

Snake Head: Row {hr}, Col {hc} (@)
Food: Row {fr}, Col {fc} (★) - {rel_vert} and {rel_horiz} ({curr_dist} steps)
Legend: '@' Head, 'o' Body, '★' Food, '.' Empty
"""

        questions = {
            "direction": choice(
                f"Based on the grid map, select the best safe move direction for the snake head (@) to reach food (★). "
                f"Food is {rel_vert} and {rel_horiz}.",
                criteria,
            ),
            "danger_level": score(
                "Evaluate the immediate survival danger of the snake's current position.",
                [
                    "Safe: multiple clear paths and ample space",
                    "Moderate: walls or body segments nearby",
                    "High danger: restricted corridor or near-fatal position",
                ],
            ),
            "food_reachable": noul(
                "Is there an open path towards the food without immediate collision?"
            ),
        }

        return state, questions, alias_to_dir

    def render_ansi(
        self,
        last_decision: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        avg_latency_ms: float | None = None,
        elapsed_s: float = 0.0,
    ) -> str:
        board_rows: list[str] = []
        body_set = set(self.snake[1:])
        head = self.head

        # Top border
        board_rows.append(f"{CLR_CYAN}┌" + "──" * self.width + "┐" + CLR_RESET)
        for r in range(self.height):
            cells = [f"{CLR_CYAN}│{CLR_RESET}"]
            for c in range(self.width):
                pos = (r, c)
                if pos == head:
                    icon = HEAD_ICONS.get(self.direction, "@")
                    cells.append(f"{CLR_BRIGHT_GREEN}{CLR_BOLD}{icon} {CLR_RESET}")
                elif pos in body_set:
                    cells.append(f"{CLR_GREEN}● {CLR_RESET}")
                elif pos == self.food:
                    cells.append(f"{CLR_RED}{CLR_BOLD}★ {CLR_RESET}")
                else:
                    cells.append(f"{CLR_GRAY}· {CLR_RESET}")
            cells.append(f"{CLR_CYAN}│{CLR_RESET}")
            board_rows.append("".join(cells))
        board_rows.append(f"{CLR_CYAN}└" + "──" * self.width + "┘" + CLR_RESET)

        # Elapsed time format: MM:SS.cs
        mins = int(elapsed_s // 60)
        secs = elapsed_s % 60
        elapsed_str = f"{mins:02d}:{secs:05.2f}"

        # Speedup vs traditional autoregressive text generation (~1,200ms)
        speedup_str = ""
        if latency_ms and latency_ms > 0:
            speedup_ratio = max(1.0, 1200.0 / latency_ms)
            speedup_str = f" (~{speedup_ratio:.0f}× faster than AR JSON)"

        # Right sidebar telemetry & decision breakdown
        sidebar: list[str] = [
            f"{CLR_BOLD}⚡ SYSTEM ONE (0.5B SFT){CLR_RESET}   {CLR_YELLOW}⏱ Elapsed: {elapsed_str}{CLR_RESET}",
            f"{CLR_DIM}─────────────────────────────────────────────────────────────────{CLR_RESET}",
            f"  Step: {CLR_YELLOW}{self.steps:<3}{CLR_RESET} | Score: {CLR_BRIGHT_GREEN}{self.score:<2}{CLR_RESET} | Length: {len(self.snake)} | Food Dist: {self.manhattan_distance(self.head, self.food)}",
        ]

        if latency_ms is not None:
            avg_str = f"{avg_latency_ms:.1f} ms" if avg_latency_ms else f"{latency_ms:.1f} ms"
            sidebar.append(
                f"  {CLR_BOLD}⚡ Latency:{CLR_RESET} {CLR_CYAN}{latency_ms:5.1f} ms{CLR_RESET} (Avg: {avg_str}){CLR_BRIGHT_GREEN}{speedup_str}{CLR_RESET}"
            )
            sidebar.append(
                f"  {CLR_DIM}   [Mechanism: Option Logits Softmax • Shared Prefix KV • No AR]{CLR_RESET}"
            )
        else:
            sidebar.append(f"  {CLR_BOLD}⚡ Latency:{CLR_RESET} Ready")

        sidebar.append(f"{CLR_BOLD}🧠 Decision Rationale & Candidates Evaluated:{CLR_RESET}")
        if last_decision:
            chosen = last_decision.get("direction", "N/A")
            conf = last_decision.get("confidence", 0.0)
            probs = last_decision.get("probabilities", {})
            analyses = last_decision.get("analyses", {})

            for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
                p = probs.get(d, 0.0)
                bar_len = int(p * 12)
                bar_str = "█" * bar_len + "░" * (12 - bar_len)
                is_picked = d == chosen
                check = f"{CLR_BRIGHT_GREEN}{CLR_BOLD}[✓]{CLR_RESET}" if is_picked else f"{CLR_DIM}[ ]{CLR_RESET}"
                d_color = CLR_BRIGHT_GREEN if is_picked else CLR_RESET
                p_color = CLR_CYAN if is_picked else CLR_GRAY

                # Lookahead summary
                analysis = analyses.get(d)
                if analysis:
                    if not analysis.is_safe:
                        status_str = f"{CLR_RED}{analysis.summary()[:34]}{CLR_RESET}"
                    elif analysis.is_closer:
                        status_str = f"{CLR_BRIGHT_GREEN}{analysis.summary()[:34]}{CLR_RESET}"
                    else:
                        status_str = f"{CLR_YELLOW}{analysis.summary()[:34]}{CLR_RESET}"
                else:
                    status_str = ""

                sidebar.append(
                    f"   {check} {d_color}{d:<5}{CLR_RESET} | {p_color}{p*100:5.1f}% [{bar_str}]{CLR_RESET} | {status_str}"
                )

            # Multi-question status
            danger = last_decision.get("danger")
            danger_str = f"{danger:.2f}/2.0" if danger is not None else "N/A"
            reach = last_decision.get("food_reachable")
            reach_str = f"{reach}" if reach is not None else "N/A"
            sidebar.append(
                f"  {CLR_BOLD}📋 Parallel Outputs:{CLR_RESET} Choice Conf: {CLR_BRIGHT_GREEN}{conf * 100:.1f}%{CLR_RESET} | Danger: {CLR_YELLOW}{danger_str}{CLR_RESET} | Reachable: {CLR_GREEN if reach else CLR_RED}{reach_str}{CLR_RESET}"
            )
        else:
            sidebar.append("   (Waiting for first decision...)")

        if self.game_over:
            sidebar.append(f"  {CLR_RED}{CLR_BOLD}💥 GAME OVER: {self.death_reason}{CLR_RESET}")

        # Combine board (left) and sidebar (right) side by side
        total_lines = max(len(board_rows), len(sidebar))
        combined: list[str] = []
        for i in range(total_lines):
            left = board_rows[i] if i < len(board_rows) else " " * (self.width * 2 + 2)
            right = sidebar[i] if i < len(sidebar) else ""
            combined.append(f"{left}  {right}")

        return "\n".join(combined)

    def render_frame_pil(
        self,
        last_decision: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        avg_latency_ms: float | None = None,
        elapsed_s: float = 0.0,
    ) -> Image.Image:
        """Render a rich graphical dashboard frame using Pillow."""
        img_w, img_h = 960, 560
        img = Image.new("RGB", (img_w, img_h), color="#0b0f19")
        draw = ImageDraw.Draw(img)

        # Fonts
        font_title = _get_font(22, bold=True)
        font_tag = _get_font(12, bold=True)
        font_sub = _get_font(12, bold=False)
        font_timer = _get_font(15, bold=True)
        font_stat_val = _get_font(20, bold=True)
        font_stat_lbl = _get_font(11, bold=False)
        font_body = _get_font(13, bold=False)
        font_bold = _get_font(13, bold=True)
        font_small = _get_font(11, bold=False)

        # 1. Left: Board
        board_box_size = 500
        pad_x, pad_y = 25, 30
        draw.rounded_rectangle(
            [pad_x - 5, pad_y - 5, pad_x + board_box_size + 5, pad_y + board_box_size + 5],
            radius=10,
            fill="#111827",
            outline="#1f2937",
            width=2,
        )

        cell_size = board_box_size // max(self.width, self.height)
        board_w = cell_size * self.width
        board_h = cell_size * self.height
        bx0 = pad_x + (board_box_size - board_w) // 2
        by0 = pad_y + (board_box_size - board_h) // 2

        # Draw grid cells
        for r in range(self.height):
            for c in range(self.width):
                x = bx0 + c * cell_size
                y = by0 + r * cell_size
                draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], fill="#0f172a", outline="#1e293b")

        # Draw food (Apple)
        if self.food != (-1, -1):
            fr, fc = self.food
            fx = bx0 + fc * cell_size + cell_size // 2
            fy = by0 + fr * cell_size + cell_size // 2
            rad = int(cell_size * 0.38)
            draw.ellipse([fx - rad, fy - rad, fx + rad, fy + rad], fill="#ef4444", outline="#b91c1c")
            draw.ellipse([fx, fy - rad - 4, fx + 5, fy - rad + 2], fill="#22c55e")

        # Draw snake body
        for idx, (br, bc) in enumerate(self.snake[1:]):
            sx = bx0 + bc * cell_size
            sy = by0 + br * cell_size
            inset = 3
            draw.rounded_rectangle(
                [sx + inset, sy + inset, sx + cell_size - inset, sy + cell_size - inset],
                radius=5,
                fill="#10b981",
                outline="#059669",
            )

        # Draw snake head
        hr, hc = self.head
        hx = bx0 + hc * cell_size
        hy = by0 + hr * cell_size
        draw.rounded_rectangle(
            [hx + 2, hy + 2, hx + cell_size - 2, hy + cell_size - 2],
            radius=7,
            fill="#34d399",
            outline="#059669",
            width=2,
        )

        # Head eyes
        eye_rad = max(2, cell_size // 10)
        center_x = hx + cell_size // 2
        center_y = hy + cell_size // 2
        eye_offset = cell_size // 4
        dr, dc = DIRECTIONS.get(self.direction, (0, 1))

        if dr == 0:
            e1 = (center_x + dc * eye_offset, center_y - eye_offset)
            e2 = (center_x + dc * eye_offset, center_y + eye_offset)
        else:
            e1 = (center_x - eye_offset, center_y + dr * eye_offset)
            e2 = (center_x + eye_offset, center_y + dr * eye_offset)

        draw.ellipse([e1[0] - eye_rad, e1[1] - eye_rad, e1[0] + eye_rad, e1[1] + eye_rad], fill="#022c22")
        draw.ellipse([e2[0] - eye_rad, e2[1] - eye_rad, e2[0] + eye_rad, e2[1] + eye_rad], fill="#022c22")

        # 2. Right: System One HUD Panel
        hx0 = 555
        hy0 = pad_y - 5
        panel_w = 380
        panel_h = board_box_size + 10

        draw.rounded_rectangle(
            [hx0, hy0, hx0 + panel_w, hy0 + panel_h],
            radius=10,
            fill="#111827",
            outline="#1f2937",
            width=2,
        )

        curr_y = hy0 + 14
        # Header with Timer
        mins = int(elapsed_s // 60)
        secs = elapsed_s % 60
        timer_text = f"⏱ {mins:02d}:{secs:05.2f}"

        draw.text((hx0 + 16, curr_y), "SYSTEM ONE", font=font_title, fill="#38bdf8")
        draw.rounded_rectangle([hx0 + 165, curr_y + 4, hx0 + 225, curr_y + 24], radius=4, fill="#6b21a8")
        draw.text((hx0 + 173, curr_y + 6), "0.5B SFT", font=font_tag, fill="#f3e8ff")
        draw.text((hx0 + panel_w - 110, curr_y + 5), timer_text, font=font_timer, fill="#facc15")
        curr_y += 28
        draw.text((hx0 + 16, curr_y), "Real-time Autonomous Decision Engine", font=font_sub, fill="#94a3b8")
        curr_y += 22
        draw.line([hx0 + 16, curr_y, hx0 + panel_w - 16, curr_y], fill="#1f2937", width=1)
        curr_y += 12

        # Stat cards: Step, Score, Length, Food Dist
        card_w = (panel_w - 32 - 18) // 3
        stats = [("STEP", str(self.steps), "#facc15"), ("SCORE", str(self.score), "#4ade80"), ("LENGTH", str(len(self.snake)), "#38bdf8")]
        for i, (lbl, val, col) in enumerate(stats):
            cx = hx0 + 16 + i * (card_w + 9)
            draw.rounded_rectangle([cx, curr_y, cx + card_w, curr_y + 44], radius=6, fill="#1e293b")
            draw.text((cx + 10, curr_y + 5), lbl, font=font_stat_lbl, fill="#94a3b8")
            draw.text((cx + 10, curr_y + 19), val, font=font_stat_val, fill=col)
        curr_y += 54

        # Latency & Speedup Banner
        draw.rounded_rectangle([hx0 + 16, curr_y, hx0 + panel_w - 16, curr_y + 50], radius=6, fill="#1e293b", outline="#334155")
        if latency_ms is not None:
            speedup = max(1.0, 1200.0 / latency_ms)
            lat_text = f"⚡ {latency_ms:.1f} ms (~{speedup:.0f}× faster than AR)"
            draw.text((hx0 + 24, curr_y + 6), lat_text, font=font_bold, fill="#facc15")
            avg_text = f"Avg: {avg_latency_ms:.1f} ms • Zero-token Option Softmax" if avg_latency_ms else "Zero-token Option Softmax"
            draw.text((hx0 + 24, curr_y + 26), avg_text, font=font_small, fill="#38bdf8")
        else:
            draw.text((hx0 + 24, curr_y + 14), "⚡ Engine Ready", font=font_bold, fill="#facc15")
        curr_y += 60

        # How Decision is Made: Candidate Directions Evaluated
        draw.text((hx0 + 16, curr_y), "Candidates Evaluated (How Decision is Made):", font=font_bold, fill="#e2e8f0")
        curr_y += 22

        if last_decision:
            chosen = last_decision.get("direction", "N/A")
            conf = last_decision.get("confidence", 0.0)
            probs = last_decision.get("probabilities", {})
            analyses = last_decision.get("analyses", {})

            for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
                p = probs.get(d, 0.0)
                is_picked = d == chosen
                row_bg = "#1e293b" if is_picked else "#0f172a"
                row_border = "#22c55e" if is_picked else "#1e293b"
                draw.rounded_rectangle([hx0 + 16, curr_y, hx0 + panel_w - 16, curr_y + 32], radius=5, fill=row_bg, outline=row_border)

                # Checkmark / label
                chk = "✓ " if is_picked else "  "
                chk_col = "#4ade80" if is_picked else "#64748b"
                draw.text((hx0 + 22, curr_y + 7), f"{chk}{d:<5}", font=font_bold if is_picked else font_body, fill=chk_col)

                # Prob bar
                bar_x = hx0 + 95
                bar_max_w = 75
                draw.rounded_rectangle([bar_x, curr_y + 11, bar_x + bar_max_w, curr_y + 19], radius=2, fill="#0b0f19")
                if p > 0:
                    fill_col = "#38bdf8" if is_picked else "#475569"
                    draw.rounded_rectangle([bar_x, curr_y + 11, bar_x + int(bar_max_w * p), curr_y + 19], radius=2, fill=fill_col)
                draw.text((bar_x + bar_max_w + 6, curr_y + 8), f"{p*100:4.1f}%", font=font_small, fill="#cbd5e1")

                # Reason / Lookahead
                analysis = analyses.get(d)
                if analysis:
                    if not analysis.is_safe:
                        r_text = "DEADLY"
                        r_col = "#f87171"
                    elif analysis.is_closer:
                        r_text = f"Closer (d:{analysis.distance_after})"
                        r_col = "#4ade80"
                    else:
                        r_text = f"Farther (d:{analysis.distance_after})"
                        r_col = "#fbbf24"
                    draw.text((hx0 + 225, curr_y + 8), r_text, font=font_small, fill=r_col)

                curr_y += 36

            curr_y += 8
            # Multi-Question outputs
            draw.text((hx0 + 16, curr_y), f"Choice Confidence: {conf * 100:.1f}%", font=font_bold, fill="#38bdf8")
            curr_y += 18
            danger = last_decision.get("danger")
            if danger is not None:
                d_desc = "Safe" if danger < 0.6 else ("Caution" if danger < 1.3 else "Danger")
                draw.text((hx0 + 16, curr_y), f"Danger Score (score): {danger:.2f}/2.0 ({d_desc})", font=font_small, fill="#fbbf24")
                curr_y += 18

            reach = last_decision.get("food_reachable")
            if reach is not None:
                r_col = "#4ade80" if reach else "#f87171"
                draw.text((hx0 + 16, curr_y), f"Food Reachable (noul): {reach}", font=font_small, fill=r_col)
                curr_y += 18

        if self.game_over:
            draw.rounded_rectangle([hx0 + 16, curr_y, hx0 + panel_w - 16, curr_y + 36], radius=6, fill="#7f1d1d")
            draw.text((hx0 + 24, curr_y + 10), f"GAME OVER: {self.death_reason[:30]}", font=font_bold, fill="#fecaca")

        return img


def run_episode(
    client: SystemOneClient,
    game: SnakeGame,
    delay: float = 0.15,
    max_steps: int = 200,
    animate: bool = True,
    gif_path: str | Path | None = None,
    fps: int = 6,
) -> dict[str, Any]:
    last_decision: dict[str, Any] | None = None
    latencies: list[float] = []
    frames: list[Image.Image] = []
    start_time = time.perf_counter()

    if animate:
        # Clear screen and hide cursor
        sys.stdout.write("\033[2J\033[H\033[?25l")
        sys.stdout.flush()

    try:
        while not game.game_over and game.steps < max_steps:
            state, questions, alias_to_dir = game.build_systemone_payload()
            analyses = game.analyze_all_directions()

            t0 = time.perf_counter()
            response = client.system_one(state=state, questions=questions)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            avg_lat = sum(latencies) / len(latencies)
            elapsed_s = time.perf_counter() - start_time

            # Parse direction choice
            dir_ans = response.answers["direction"]
            chosen_alias = dir_ans.choice
            chosen_dir = alias_to_dir.get(chosen_alias, game.direction)

            # Map probabilities back to direction names
            raw_probs = dir_ans.probabilities or {}
            named_probs = {
                alias_to_dir.get(alias, alias): p for alias, p in raw_probs.items()
            }

            danger_val = response.answers.get("danger_level")
            reach_val = response.answers.get("food_reachable")

            last_decision = {
                "direction": chosen_dir,
                "confidence": dir_ans.confidence or 0.0,
                "probabilities": named_probs,
                "danger": danger_val.score if danger_val else None,
                "food_reachable": reach_val.noul if reach_val else None,
                "analyses": analyses,
            }

            if gif_path:
                frames.append(game.render_frame_pil(last_decision, elapsed_ms, avg_lat, elapsed_s))

            if animate:
                # Move cursor to home and redraw
                sys.stdout.write("\033[H")
                sys.stdout.write(game.render_ansi(last_decision, elapsed_ms, avg_lat, elapsed_s) + "\n")
                sys.stdout.flush()
                if delay > 0:
                    time.sleep(delay)

            game.step(chosen_dir)

        # Final render
        total_elapsed_s = time.perf_counter() - start_time
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        if gif_path:
            frames.append(game.render_frame_pil(last_decision, latencies[-1] if latencies else 0.0, avg_lat, total_elapsed_s))

        if animate:
            sys.stdout.write("\033[H")
            sys.stdout.write(
                game.render_ansi(last_decision, latencies[-1] if latencies else 0.0, avg_lat, total_elapsed_s) + "\n"
            )
            sys.stdout.flush()

    finally:
        if animate:
            # Restore cursor
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()

    if gif_path and frames:
        out_p = Path(gif_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Duplicate last frame 3 times to freeze slightly on end screen
        extended_frames = frames + [frames[-1]] * 4
        extended_frames[0].save(
            out_p,
            save_all=True,
            append_images=extended_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "score": game.score,
        "steps": game.steps,
        "length": len(game.snake),
        "death_reason": game.death_reason or ("Max steps reached" if game.steps >= max_steps else "Alive"),
        "avg_latency_ms": avg_latency,
        "total_decisions": len(latencies),
        "gif_saved": str(gif_path) if (gif_path and frames) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Snake game demo powered by systemone-lite."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="systemone-lite-latest",
        help="Model ID or alias (default: systemone-lite-latest / Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use deterministic StubEngine (no weight download or GPU required)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Remote or local FastAPI server URL (e.g. http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=10,
        help="Grid width (default: 10)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=10,
        help="Grid height (default: 10)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.15,
        help="Delay in seconds between steps for visualization (default: 0.15)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for food generation",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable animated display (useful for batch benchmark)",
    )
    parser.add_argument(
        "--gif",
        type=str,
        default=None,
        help="Path to save animated GIF (e.g. benchmarks/viral/snake_demo.gif)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Frames per second for saved GIF (default: 6)",
    )
    args = parser.parse_args()

    # Initialize client
    if args.stub:
        engine = StubEngine()
        client = SystemOneClient(engine=engine)
        print(f"{CLR_CYAN}[Info] Running with deterministic StubEngine{CLR_RESET}")
    else:
        client = SystemOneClient(base_url=args.base_url, model=args.model)
        if args.base_url:
            print(f"{CLR_CYAN}[Info] Connecting to System One API at {args.base_url}{CLR_RESET}")
        else:
            print(f"{CLR_CYAN}[Info] Running in-process with model '{args.model}'{CLR_RESET}")

    results: list[dict[str, Any]] = []
    animate = not args.quiet and args.episodes == 1

    for ep in range(1, args.episodes + 1):
        ep_seed = (args.seed + ep - 1) if args.seed is not None else None
        game = SnakeGame(width=args.width, height=args.height, seed=ep_seed)
        if not animate:
            print(f"Running episode {ep}/{args.episodes}...", end="", flush=True)

        res = run_episode(
            client=client,
            game=game,
            delay=args.delay if animate else 0.0,
            max_steps=args.max_steps,
            animate=animate,
            gif_path=args.gif if ep == 1 else None,
            fps=args.fps,
        )
        results.append(res)

        if not animate:
            print(
                f" Done. Score: {res['score']}, Steps: {res['steps']}, "
                f"Latency: {res['avg_latency_ms']:.1f}ms, End: {res['death_reason']}"
            )
        if res.get("gif_saved"):
            print(f"{CLR_GREEN}✔ Saved gameplay GIF to {res['gif_saved']}{CLR_RESET}")

    # Print summary if multiple episodes or quiet mode
    if args.episodes > 1 or args.quiet:
        avg_score = sum(r["score"] for r in results) / len(results)
        avg_steps = sum(r["steps"] for r in results) / len(results)
        avg_lat = sum(r["avg_latency_ms"] for r in results) / len(results)
        print("\n" + "=" * 50)
        print(f"{CLR_BOLD}Snake AI Evaluation Summary ({args.episodes} episodes){CLR_RESET}")
        print("=" * 50)
        print(f"Average Score:        {CLR_BRIGHT_GREEN}{avg_score:.2f}{CLR_RESET}")
        print(f"Average Steps:        {CLR_YELLOW}{avg_steps:.1f}{CLR_RESET}")
        print(f"Average Latency:      {CLR_CYAN}{avg_lat:.2f} ms{CLR_RESET}")
        print("=" * 50)


if __name__ == "__main__":
    main()
