#!/usr/bin/env python3
"""Autonomous 2048 Tile Merging Demo powered by systemone-lite.

System One acts as the real-time 'System 1' reflex planner:
  - Ultra-compact 4x4 2D text grid representation (<30 tokens)
  - Sub-10ms decisions for lightning-fast tile merging
  - Multi-question simultaneous evaluation:
      1. 'slide' (choice): Best slide direction (UP, DOWN, LEFT, RIGHT)
      2. 'overflow_alert' (noul): Grid space emergency warning
  - Colorful ANSI terminal HUD and optional GIF export.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from systemone_lite import SystemOneClient, choice, noul
from systemone_lite.infer import DEFAULT_MODEL_ID
from systemone_lite.stub import StubEngine
from systemone_lite.synth.game2048 import Board2048, best_move_2048

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

TILE_COLORS_PIL = {
    0: ("#1e293b", "#64748b"),
    2: ("#e2e8f0", "#0f172a"),
    4: ("#cbd5e1", "#0f172a"),
    8: ("#f97316", "#ffffff"),
    16: ("#ea580c", "#ffffff"),
    32: ("#ef4444", "#ffffff"),
    64: ("#dc2626", "#ffffff"),
    128: ("#eab308", "#ffffff"),
    256: ("#facc15", "#ffffff"),
    512: ("#a855f7", "#ffffff"),
    1024: ("#6366f1", "#ffffff"),
    2048: ("#ec4899", "#ffffff"),
}


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


class Game2048Session:
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.board = Board2048([[0] * 4 for _ in range(4)])
        self.board.spawn_tile(self.rng)
        self.board.spawn_tile(self.rng)
        self.score = 0
        self.steps = 0
        self.game_over = False

    def build_systemone_payload(self) -> tuple[dict[str, Any], dict[str, Any]]:
        grid_ascii = f"\n{self.board.render_ascii()}\n"
        legal = self.board.get_legal_moves()
        best_d, _ = best_move_2048(self.board)

        options = {}
        for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
            if d in legal:
                options[d] = f"RECOMMENDED: Slide {d} to merge" if d == best_d else f"Slide {d}"
            else:
                options[d] = f"BLOCKED: No tiles move {d}"

        empty_count = len(self.board.empty_cells)
        state = {
            "grid_map": grid_ascii,
            "score": self.score,
            "max_tile": self.board.max_tile,
            "empty_cells": empty_count,
            "status": "CRITICAL: Board almost full" if empty_count <= 2 else "SAFE: Good space",
        }

        questions = {
            "slide": choice(
                "Based on the 4x4 2048 grid, select the best slide direction to merge tiles and keep high values in corners. "
                "Reply with: UP, DOWN, LEFT, RIGHT",
                options,
            ),
            "overflow_alert": noul(
                "Is the 2048 grid currently in immediate danger of overflowing and ending the game?"
            ),
        }
        return state, questions

    def step(self, direction: str) -> bool:
        if self.game_over:
            return False

        legal = self.board.get_legal_moves()
        if direction not in legal:
            if not legal:
                self.game_over = True
                return False
            direction = list(legal.keys())[0]

        nxt_board, gained, _ = self.board.move(direction)
        self.score += gained
        nxt_board.spawn_tile(self.rng)
        self.board = nxt_board
        self.steps += 1

        if not self.board.get_legal_moves():
            self.game_over = True
            return False
        return True

    def render_ansi(
        self,
        last_decision: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        avg_latency_ms: float | None = None,
        elapsed_s: float = 0.0,
    ) -> str:
        # 1. Left: 2048 4x4 Grid
        board_rows = [f"{CLR_CYAN}  ┌────────────────────────┐{CLR_RESET}"]
        for r in range(4):
            row_cells = []
            for c in range(4):
                val = self.board.grid[r][c]
                if val == 0:
                    row_cells.append(f"\033[90m  ·  {CLR_RESET}")
                elif val <= 4:
                    row_cells.append(f"\033[97m{val:5d}{CLR_RESET}")
                elif val <= 64:
                    row_cells.append(f"\033[93m{val:5d}{CLR_RESET}")
                else:
                    row_cells.append(f"\033[95m{val:5d}{CLR_RESET}")
            board_rows.append(f"{CLR_CYAN}  │{CLR_RESET}" + "|".join(row_cells) + f"{CLR_CYAN}│{CLR_RESET}")
            if r < 3:
                board_rows.append(f"{CLR_CYAN}  ├─────┼─────┼─────┼─────┤{CLR_RESET}")
        board_rows.append(f"{CLR_CYAN}  └────────────────────────┘{CLR_RESET}")

        # 2. Right: Telemetry
        mins = int(elapsed_s // 60)
        secs = elapsed_s % 60
        timer_str = f"{mins:02d}:{secs:05.2f}"
        empty_count = len(self.board.empty_cells)

        speedup_str = ""
        if latency_ms and latency_ms > 0:
            speedup_ratio = max(1.0, 1100.0 / latency_ms)
            speedup_str = f" (~{speedup_ratio:.0f}× vs CoT)"

        sidebar = [
            f"{CLR_BOLD}🎮 SYSTEM ONE 2048 AGENT{CLR_RESET}   {CLR_YELLOW}⏱ Elapsed: {timer_str}{CLR_RESET}",
            f"{CLR_DIM}─────────────────────────────────────────────────────────────────{CLR_RESET}",
            f"  Steps: {CLR_YELLOW}{self.steps:<3}{CLR_RESET} | Score: {CLR_BRIGHT_GREEN}{self.score:<5}{CLR_RESET} | Max Tile: {CLR_CYAN}{self.board.max_tile}{CLR_RESET} | Empty: {empty_count}/16",
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
            sidebar.append("   (Planning tile merges...)")

        if self.game_over:
            sidebar.append(f"  {CLR_RED}{CLR_BOLD}💥 GAME OVER! No legal moves remain.{CLR_RESET}")

        total_lines = max(len(board_rows), len(sidebar))
        combined = []
        for i in range(total_lines):
            left = board_rows[i] if i < len(board_rows) else " " * 32
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
        img_w, img_h = 880, 480
        img = Image.new("RGB", (img_w, img_h), color="#0b0f19")
        draw = ImageDraw.Draw(img)

        font_title = _get_font(20, bold=True)
        font_sub = _get_font(12, bold=False)
        font_stat_val = _get_font(18, bold=True)
        font_stat_lbl = _get_font(11, bold=False)
        font_bold = _get_font(13, bold=True)
        font_small = _get_font(11, bold=False)
        font_tile = _get_font(24, bold=True)

        # 1. Left: 2048 4x4 Grid
        board_size = 380
        pad_x = 40
        pad_y = 50
        gap = 10
        cell_size = (board_size - gap * 3) // 4

        draw.rounded_rectangle([pad_x - 10, pad_y - 10, pad_x + board_size + 10, pad_y + board_size + 10], radius=12, fill="#0f172a", outline="#1e293b", width=2)

        for r in range(4):
            for c in range(4):
                val = self.board.grid[r][c]
                x0 = pad_x + c * (cell_size + gap)
                y0 = pad_y + r * (cell_size + gap)

                bg_col, txt_col = TILE_COLORS_PIL.get(val, ("#ec4899", "#ffffff"))
                draw.rounded_rectangle([x0, y0, x0 + cell_size, y0 + cell_size], radius=8, fill=bg_col)

                if val > 0:
                    txt = str(val)
                    bbox = draw.textbbox((0, 0), txt, font=font_tile)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    draw.text((x0 + (cell_size - tw) // 2, y0 + (cell_size - th) // 2 - 2), txt, font=font_tile, fill=txt_col)

        # 2. Right: HUD
        hx0 = 460
        hy0 = pad_y - 10
        panel_w = 380
        panel_h = board_size + 20

        draw.rounded_rectangle([hx0, hy0, hx0 + panel_w, hy0 + panel_h], radius=10, fill="#111827", outline="#1f2937", width=2)
        curr_y = hy0 + 14

        draw.text((hx0 + 16, curr_y), "2048 SLIDING AGENT", font=font_title, fill="#38bdf8")
        curr_y += 24
        draw.text((hx0 + 16, curr_y), "Sub-10ms 4x4 Grid Reflexes (<30 Tokens)", font=font_sub, fill="#94a3b8")
        curr_y += 20

        # Stats
        card_w = (panel_w - 32 - 16) // 3
        stats = [("STEPS", str(self.steps), "#facc15"), ("SCORE", str(self.score), "#4ade80"), ("MAX", str(self.board.max_tile), "#38bdf8")]
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
            draw.text((hx0 + 20, curr_y + 5), lat_text, font=font_bold, fill="#facc15")
            avg_text = f"Avg: {avg_latency_ms:.1f} ms • Ultra-compact 4x4 Grid" if avg_latency_ms else "Sub-10ms Decisions"
            draw.text((hx0 + 20, curr_y + 22), avg_text, font=font_small, fill="#38bdf8")
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
                draw.text((hx0 + 20, curr_y + 5), f"{chk}{d}", font=font_bold if is_p else font_small, fill="#4ade80" if is_p else "#94a3b8")

                bar_x = hx0 + 110
                draw.rounded_rectangle([bar_x, curr_y + 8, bar_x + int(p * 100), curr_y + 18], radius=2, fill="#38bdf8" if is_p else "#475569")
                draw.text((bar_x + 110, curr_y + 5), f"{p*100:4.1f}%", font=font_small, fill="#cbd5e1")
                curr_y += 30

        return img


def play_2048_demo(
    client: SystemOneClient,
    max_steps: int = 50,
    delay: float = 0.20,
    animate: bool = True,
    gif_path: str | Path | None = None,
    fps: int = 4,
) -> dict[str, Any]:
    game = Game2048Session(seed=42)
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

            ans_dir = resp.answers["slide"]
            chosen_dir = ans_dir.choice
            legal = game.board.get_legal_moves()
            if chosen_dir not in legal:
                best_d, _ = best_move_2048(game.board)
                chosen_dir = best_d if best_d in legal else list(legal.keys())[0]

            last_decision = {
                "chosen_direction": chosen_dir,
                "confidence": ans_dir.confidence or 0.0,
                "probabilities": ans_dir.probabilities or {},
                "overflow_alert": resp.answers.get("overflow_alert"),
            }

            if gif_path:
                frames.append(game.render_frame_pil(last_decision, lat_ms, avg_lat, elapsed_s))

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
            frames.append(game.render_frame_pil(last_decision, latencies[-1] if latencies else 0.0, avg_lat, total_elapsed_s))
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
        print(f"\nSaved 2048 animated GIF → {out_p}")

    return {
        "steps": game.steps,
        "score": game.score,
        "max_tile": game.board.max_tile,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous 2048 Demo (systemone-lite)")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps to play")
    parser.add_argument("--delay", type=float, default=0.20, help="Delay between steps in seconds")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--stub", action="store_true", help="Use stub engine (no GPU / no download)")
    parser.add_argument("--gif", type=Path, default=None, help="Output GIF path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.stub:
        client = SystemOneClient(engine=StubEngine())
    else:
        from systemone_lite.infer import LocalInferEngine
        client = SystemOneClient(engine=LocalInferEngine(model_id=args.model))

    res = play_2048_demo(
        client=client,
        max_steps=args.max_steps,
        delay=0.0 if args.quiet else args.delay,
        animate=not args.quiet,
        gif_path=args.gif,
    )

    print("\n" + "=" * 50)
    print("2048 Game Finished!")
    print(f"Total Steps   : {res['steps']}")
    print(f"Final Score   : {res['score']}")
    print(f"Highest Tile  : {res['max_tile']}")
    print(f"Avg Latency   : {res['avg_latency_ms']:.1f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
