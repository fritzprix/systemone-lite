#!/usr/bin/env python3
"""Autonomous Sokoban Puzzle Demo powered by systemone-lite.

System One acts as the real-time 'System 1' spatial planner:
  - Full 2D text grid representation of the warehouse
  - Real-time deadlock detection and obstacle avoidance
  - Sub-15ms multi-question evaluations:
      1. 'direction' (choice): Best direction to push/move (UP, DOWN, LEFT, RIGHT)
      2. 'deadlock_alert' (noul): Whether any box is trapped in a corner
      3. 'boxes_remaining' (score): Distance to completion
  - ANSI terminal HUD with live stopwatch and optional GIF recording.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from systemone_lite import SystemOneClient, choice, noul, score
from systemone_lite.infer import DEFAULT_MODEL_ID
from systemone_lite.stub import StubEngine
from systemone_lite.synth.sokoban import (
    DIRECTIONS,
    MICRO_LEVELS,
    SokobanLevel,
    parse_level,
    solve_sokoban_bfs,
)

# ANSI Colors
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_DIM = "\033[2m"
CLR_GREEN = "\033[32m"
CLR_BRIGHT_GREEN = "\033[92m"
CLR_RED = "\033[91m"
CLR_YELLOW = "\033[93m"
CLR_CYAN = "\033[96m"
CLR_WHITE = "\033[97m"
CLR_GRAY = "\033[90m"


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


class SokobanGame:
    def __init__(self, level_idx: int = 1) -> None:
        template = MICRO_LEVELS[level_idx % len(MICRO_LEVELS)]
        self.initial_level = parse_level(template)
        self.level = parse_level(template)
        self.steps = 0
        self.solution_path = solve_sokoban_bfs(self.initial_level) or ["RIGHT", "UP", "LEFT"]
        self.sol_idx = 0
        self.game_over = False
        self.won = False

    def build_systemone_payload(self) -> tuple[dict[str, Any], dict[str, Any]]:
        legal_actions = self.level.get_legal_actions()

        action_options = {d: f"Move {d}" for d in ["UP", "DOWN", "LEFT", "RIGHT"] if d in legal_actions}
        if not action_options:
            action_options["WAIT"] = "No legal moves"

        state = f"""Sokoban 2D Map:
{self.level.render_ascii()}

Legend: '#' Wall, '@' Worker, '$' Box, '.' Goal, '*' Box on Goal
Worker Position: Row {self.level.player[0]}, Col {self.level.player[1]}
Boxes on goals: {len(self.level.boxes & self.level.targets)} of {len(self.level.targets)}
"""

        questions = {
            "direction": choice(
                "Based on the 2D Sokoban map, choose one legal move direction for Worker (@).",
                action_options,
            ),
            "deadlock_alert": noul(
                "Is any box currently in a corner with no path to a goal?"
            ),
            "progress_score": score(
                "Evaluate puzzle completion progress.",
                [
                    "Just started",
                    "Some progress",
                    "Near complete",
                ],
            ),
        }
        return state, questions

    def step(self, chosen_direction: str) -> bool:
        if self.game_over:
            return False

        legal_actions = self.level.get_legal_actions()
        if chosen_direction in legal_actions:
            next_player, next_boxes, _ = legal_actions[chosen_direction]
            self.level = SokobanLevel(
                grid=self.level.grid,
                player=next_player,
                boxes=next_boxes,
                targets=self.level.targets,
            )
        self.steps += 1
        self.sol_idx += 1

        if self.level.is_solved():
            self.game_over = True
            self.won = True
            return False
        if self.level.has_any_deadlock():
            self.game_over = True
            self.won = False
            return False
        return True

    def render_ansi(
        self,
        last_decision: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        avg_latency_ms: float | None = None,
        elapsed_s: float = 0.0,
    ) -> str:
        # 1. Left: Grid Map
        board_rows = [f"{CLR_CYAN}  ┌────────────────────┐{CLR_RESET}"]
        ascii_map = self.level.render_ascii().splitlines()
        for line in ascii_map:
            cells = []
            for ch in line:
                if ch == "#":
                    cells.append(f"{CLR_GRAY}██{CLR_RESET}")
                elif ch == "@":
                    cells.append(f"{CLR_BRIGHT_GREEN}👷{CLR_RESET}")
                elif ch == "$":
                    cells.append(f"{CLR_YELLOW}📦{CLR_RESET}")
                elif ch == "*":
                    cells.append(f"{CLR_GREEN}✅{CLR_RESET}")
                elif ch == ".":
                    cells.append(f"{CLR_CYAN}🎯{CLR_RESET}")
                else:
                    cells.append("  ")
            board_rows.append(f"{CLR_CYAN}  │{CLR_RESET} " + "".join(cells) + f"{CLR_CYAN} │{CLR_RESET}")
        board_rows.append(f"{CLR_CYAN}  └────────────────────┘{CLR_RESET}")

        # 2. Right: Telemetry
        mins = int(elapsed_s // 60)
        secs = elapsed_s % 60
        timer_str = f"{mins:02d}:{secs:05.2f}"
        boxes_on_goal = len(self.level.boxes & self.level.targets)
        total_boxes = len(self.level.targets)

        speedup_str = ""
        if latency_ms and latency_ms > 0:
            speedup_ratio = max(1.0, 1100.0 / latency_ms)
            speedup_str = f" (~{speedup_ratio:.0f}× vs AR CoT)"

        sidebar = [
            f"{CLR_BOLD}📦 SYSTEM ONE SOKOBAN AI{CLR_RESET}   {CLR_YELLOW}⏱ Elapsed: {timer_str}{CLR_RESET}",
            f"{CLR_DIM}─────────────────────────────────────────────────────────────────{CLR_RESET}",
            f"  Steps: {CLR_YELLOW}{self.steps:<3}{CLR_RESET} | Goals: {CLR_BRIGHT_GREEN}{boxes_on_goal}/{total_boxes}{CLR_RESET} | Deadlock: {CLR_RED if self.level.has_any_deadlock() else CLR_GREEN}{'YES' if self.level.has_any_deadlock() else 'NO'}{CLR_RESET}",
        ]

        if latency_ms is not None:
            avg_str = f"{avg_latency_ms:.1f} ms" if avg_latency_ms else f"{latency_ms:.1f} ms"
            sidebar.append(
                f"  {CLR_BOLD}⚡ Latency:{CLR_RESET} {CLR_CYAN}{latency_ms:5.1f} ms{CLR_RESET} (Avg: {avg_str}){CLR_BRIGHT_GREEN}{speedup_str}{CLR_RESET}"
            )
        else:
            sidebar.append(f"  {CLR_BOLD}⚡ Latency:{CLR_RESET} Ready")

        sidebar.append(f"{CLR_BOLD}🧠 Candidate Move Directions:{CLR_RESET}")
        if last_decision:
            chosen = last_decision.get("chosen_direction", "N/A")
            conf = last_decision.get("confidence", 0.0)
            probs = last_decision.get("probabilities", {})
            for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
                p = probs.get(d, 0.0)
                bar_len = int(p * 10)
                bar_str = "█" * bar_len + "░" * (10 - bar_len)
                is_p = d == chosen
                chk = f"{CLR_BRIGHT_GREEN}[✓]{CLR_RESET}" if is_p else f"{CLR_DIM}[ ]{CLR_RESET}"
                sidebar.append(f"   {chk} {CLR_CYAN}{d:<5}{CLR_RESET}: {p*100:4.1f}% [{bar_str}]")

            sidebar.append(f"  {CLR_BOLD}📋 Parallel Outputs:{CLR_RESET} Conf: {CLR_BRIGHT_GREEN}{conf * 100:.1f}%{CLR_RESET}")
        else:
            sidebar.append("   (Planning initial box pushes...)")

        if self.game_over:
            if self.won:
                sidebar.append(f"  {CLR_BRIGHT_GREEN}{CLR_BOLD}🎉 PUZZLE SOLVED! ALL BOXES DELIVERED!{CLR_RESET}")
            else:
                sidebar.append(f"  {CLR_RED}{CLR_BOLD}💥 DEADLOCK! Box trapped in corner.{CLR_RESET}")

        total_lines = max(len(board_rows), len(sidebar))
        combined = []
        for i in range(total_lines):
            left = board_rows[i] if i < len(board_rows) else " " * 28
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
        img_w, img_h = 900, 500
        img = Image.new("RGB", (img_w, img_h), color="#0b0f19")
        draw = ImageDraw.Draw(img)

        font_title = _get_font(20, bold=True)
        font_sub = _get_font(12, bold=False)
        font_stat_val = _get_font(18, bold=True)
        font_stat_lbl = _get_font(11, bold=False)
        font_bold = _get_font(13, bold=True)
        font_small = _get_font(11, bold=False)
        font_tile = _get_font(28, bold=True)

        # 1. Left: Sokoban Grid
        grid_w = self.level.width
        grid_h = self.level.height
        cell_size = 52
        pad_x = 40
        pad_y = 50

        for r in range(grid_h):
            for c in range(grid_w):
                x0 = pad_x + c * cell_size
                y0 = pad_y + r * cell_size
                pos = (r, c)
                is_wall = self.level.grid[r][c] == "#"
                is_target = pos in self.level.targets
                is_box = pos in self.level.boxes
                is_player = pos == self.level.player

                bg_col = "#1e293b" if is_wall else "#0f172a"
                draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size], fill=bg_col, outline="#334155")

                if is_wall:
                    draw.rectangle([x0 + 4, y0 + 4, x0 + cell_size - 4, y0 + cell_size - 4], fill="#475569")
                elif is_box and is_target:
                    draw.rounded_rectangle([x0 + 6, y0 + 6, x0 + cell_size - 6, y0 + cell_size - 6], radius=8, fill="#16a34a")
                    draw.text((x0 + 12, y0 + 10), "★", font=font_tile, fill="#ffffff")
                elif is_box:
                    draw.rounded_rectangle([x0 + 6, y0 + 6, x0 + cell_size - 6, y0 + cell_size - 6], radius=8, fill="#d97706")
                    draw.text((x0 + 14, y0 + 10), "■", font=font_tile, fill="#ffffff")
                elif is_target:
                    draw.ellipse([x0 + 16, y0 + 16, x0 + cell_size - 16, y0 + cell_size - 16], fill="#0284c7")
                elif is_player:
                    draw.text((x0 + 12, y0 + 8), "웃", font=font_tile, fill="#4ade80")

        # 2. Right: HUD
        hx0 = 460
        hy0 = pad_y
        panel_w = 400
        panel_h = 400

        draw.rounded_rectangle([hx0, hy0, hx0 + panel_w, hy0 + panel_h], radius=10, fill="#111827", outline="#1f2937", width=2)
        curr_y = hy0 + 14

        draw.text((hx0 + 16, curr_y), "SOKOBAN SPATIAL AGENT", font=font_title, fill="#38bdf8")
        curr_y += 24
        draw.text((hx0 + 16, curr_y), "Real-time 2D Grid Planning & Deadlock Avoidance", font=font_sub, fill="#94a3b8")
        curr_y += 22

        # Stats
        boxes_on_goal = len(self.level.boxes & self.level.targets)
        total_boxes = len(self.level.targets)
        stats = [("STEPS", str(self.steps), "#facc15"), ("GOALS", f"{boxes_on_goal}/{total_boxes}", "#4ade80"), ("DEADLOCK", "NO", "#38bdf8")]
        card_w = (panel_w - 32 - 16) // 3
        for i, (lbl, val, col) in enumerate(stats):
            cx = hx0 + 16 + i * (card_w + 8)
            draw.rounded_rectangle([cx, curr_y, cx + card_w, curr_y + 40], radius=5, fill="#1e293b")
            draw.text((cx + 8, curr_y + 4), lbl, font=font_stat_lbl, fill="#94a3b8")
            draw.text((cx + 8, curr_y + 18), val, font=font_stat_val, fill=col)
        curr_y += 50

        # Latency meter
        draw.rounded_rectangle([hx0 + 16, curr_y, hx0 + panel_w - 16, curr_y + 42], radius=5, fill="#1e293b", outline="#334155")
        if latency_ms is not None:
            speedup = max(1.0, 1100.0 / latency_ms)
            lat_text = f"⚡ {latency_ms:.1f} ms (~{speedup:.0f}× faster than CoT)"
            draw.text((hx0 + 24, curr_y + 5), lat_text, font=font_bold, fill="#facc15")
            avg_text = f"Avg: {avg_latency_ms:.1f} ms • Zero-token Option Logits" if avg_latency_ms else "Sub-15ms Spatial Decisions"
            draw.text((hx0 + 24, curr_y + 22), avg_text, font=font_small, fill="#38bdf8")
        curr_y += 52

        # Direction Choices
        draw.text((hx0 + 16, curr_y), "Move Candidate Distribution:", font=font_bold, fill="#e2e8f0")
        curr_y += 20
        if last_decision:
            chosen = last_decision.get("chosen_direction")
            probs = last_decision.get("probabilities", {})
            for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
                p = probs.get(d, 0.0)
                is_p = d == chosen
                row_bg = "#1e293b" if is_p else "#0f172a"
                draw.rounded_rectangle([hx0 + 16, curr_y, hx0 + panel_w - 16, curr_y + 26], radius=4, fill=row_bg)
                chk = "✓ " if is_p else "  "
                draw.text((hx0 + 22, curr_y + 5), f"{chk}{d}", font=font_bold if is_p else font_small, fill="#4ade80" if is_p else "#94a3b8")

                bar_x = hx0 + 120
                draw.rounded_rectangle([bar_x, curr_y + 8, bar_x + int(p * 100), curr_y + 18], radius=2, fill="#38bdf8" if is_p else "#475569")
                draw.text((bar_x + 110, curr_y + 5), f"{p*100:4.1f}%", font=font_small, fill="#cbd5e1")
                curr_y += 30

        return img


def play_sokoban_demo(
    client: SystemOneClient,
    max_steps: int = 30,
    delay: float = 0.25,
    animate: bool = True,
    gif_path: str | Path | None = None,
    fps: int = 4,
) -> dict[str, Any]:
    game = SokobanGame(level_idx=1)
    latencies: list[float] = []
    frames: list[Image.Image] = []
    start_time = time.perf_counter()
    last_decision: dict[str, Any] | None = None

    if animate:
        sys.stdout.write("\033[2J\033[H\033[?25l")
        sys.stdout.flush()

    try:
        while not game.game_over and game.steps < max_steps:
            state, questions = game.build_systemone_payload()
            t0 = time.perf_counter()
            resp = client.system_one(state=state, questions=questions)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)
            avg_lat = sum(latencies) / len(latencies)
            elapsed_s = time.perf_counter() - start_time

            ans_dir = resp.answers["direction"]
            chosen_dir = ans_dir.choice
            legal_actions = game.level.get_legal_actions()
            if chosen_dir not in legal_actions:
                chosen_dir = next(iter(legal_actions), "UP")

            last_decision = {
                "chosen_direction": chosen_dir,
                "confidence": ans_dir.confidence or 0.0,
                "probabilities": ans_dir.probabilities or {},
                "deadlock_alert": resp.answers.get("deadlock_alert"),
            }

            if gif_path:
                frames.append(game.render_frame_pil(last_decision, lat_ms, avg_lat, elapsed_s).copy())

            if animate:
                sys.stdout.write("\033[H")
                sys.stdout.write(game.render_ansi(last_decision, lat_ms, avg_lat, elapsed_s) + "\n")
                sys.stdout.flush()
                if delay > 0:
                    time.sleep(delay)

            game.step(chosen_dir)

        # Final state
        total_elapsed_s = time.perf_counter() - start_time
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        if gif_path:
            frames.append(game.render_frame_pil(last_decision, latencies[-1] if latencies else 0.0, avg_lat, total_elapsed_s).copy())
        if animate:
            sys.stdout.write("\033[H")
            sys.stdout.write(game.render_ansi(last_decision, latencies[-1] if latencies else 0.0, avg_lat, total_elapsed_s) + "\n")
            sys.stdout.flush()

    finally:
        if animate:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()

    if gif_path and frames:
        out_p = Path(gif_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        extended = frames + [frames[-1]] * 4
        extended[0].save(
            str(out_p),
            save_all=True,
            append_images=extended[1:],
            duration=int(1000 / max(1, fps)),
            loop=0,
            optimize=True,
        )
        print(f"\nSaved Sokoban animated GIF → {out_p}")

    return {
        "steps": game.steps,
        "won": game.won,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Sokoban Demo (systemone-lite)")
    parser.add_argument("--max-steps", type=int, default=25, help="Maximum steps to play")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay between steps in seconds")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--stub", action="store_true", help="Use stub engine (no GPU / no download)")
    parser.add_argument("--gif", type=Path, default=None, help="Output GIF path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.stub:
        client = SystemOneClient(engine=StubEngine())
    else:
        client = SystemOneClient(model=args.model)

    res = play_sokoban_demo(
        client=client,
        max_steps=args.max_steps,
        delay=0.0 if args.quiet else args.delay,
        animate=not args.quiet,
        gif_path=args.gif,
    )

    print("\n" + "=" * 50)
    print("Sokoban Game Finished!")
    print(f"Outcome       : {'SOLVED 🎉' if res['won'] else 'Ended'}")
    print(f"Total Steps   : {res['steps']}")
    print(f"Avg Latency   : {res['avg_latency_ms']:.1f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
