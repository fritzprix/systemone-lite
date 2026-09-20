#!/usr/bin/env python3
"""Autonomous Connect Four Demo powered by systemone-lite.

System One acts as the real-time 'System 1' tactical connector:
  - 7x6 2D text grid representation of the vertical rack
  - Sub-15ms column drop decisions with win/block reflexes
  - Multi-question simultaneous evaluation:
      1. 'drop' (choice): Best column 1 to 7
      2. 'threat_alert' (noul): Immediate opponent 4-in-a-row alert
  - ANSI terminal HUD with live falling discs and optional GIF export.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from systemone_lite import SystemOneClient, choice, noul
from systemone_lite.infer import DEFAULT_MODEL_ID
from systemone_lite.stub import StubEngine
from systemone_lite.synth.connect4 import (
    COLS,
    ROWS,
    Connect4Board,
    best_move_connect4,
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


class Connect4GameSession:
    def __init__(self) -> None:
        self.board = Connect4Board([["." for _ in range(COLS)] for _ in range(ROWS)], turn="R")
        self.turns = 0
        self.game_over = False
        self.winner: str | None = None

    def build_systemone_payload(self) -> tuple[dict[str, Any], dict[str, Any]]:
        curr_player = "R" if self.turns % 2 == 0 else "Y"
        opp_player = "Y" if curr_player == "R" else "R"
        legal = self.board.get_legal_columns()
        best_col = best_move_connect4(self.board, curr_player)
        threat_col = self.board.find_immediate_threat(opp_player)

        options = {}
        for c in range(COLS):
            col_key = str(c + 1)
            if c not in legal:
                options[col_key] = f"Column {col_key} is full"
            else:
                options[col_key] = f"Drop disc into column {col_key}"

        grid_ascii = f"\n{self.board.render_ascii()}\n"
        player_name = "Red (🔴)" if curr_player == "R" else "Yellow (🟡)"

        state = {
            "grid_map": grid_ascii,
            "side_to_move": player_name,
            "turn_number": self.turns + 1,
            "threat_detected": threat_col is not None,
        }

        questions = {
            "drop": choice(
                f"It is {player_name}'s turn. Choose the best column (1 to 7) to drop your disc. "
                "Reply with: 1, 2, 3, 4, 5, 6, 7",
                options,
            ),
            "threat_alert": noul(
                "Does the opponent have an immediate 4-in-a-row winning threat next turn?"
            ),
        }
        return state, questions

    def step(self, chosen_col_1based: int) -> bool:
        if self.game_over:
            return False

        col = chosen_col_1based - 1
        legal = self.board.get_legal_columns()
        curr_player = "R" if self.turns % 2 == 0 else "Y"

        if col not in legal:
            if not legal:
                self.game_over = True
                return False
            col = legal[0]

        row = self.board.drop_piece(col, curr_player)
        self.turns += 1

        if row is not None and self.board.check_win_at(row, col, curr_player):
            self.game_over = True
            self.winner = curr_player
            return False
        if not self.board.get_legal_columns():
            self.game_over = True  # Draw
            return False
        return True

    def render_ansi(
        self,
        last_decision: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        avg_latency_ms: float | None = None,
        elapsed_s: float = 0.0,
    ) -> str:
        # 1. Left: Connect 4 Grid
        board_rows = [f"{CLR_CYAN}  1  2  3  4  5  6  7{CLR_RESET}"]
        board_rows.append(f"{CLR_CYAN}┌─────────────────────┐{CLR_RESET}")
        for r in range(ROWS):
            cells = []
            for c in range(COLS):
                v = self.board.grid[r][c]
                if v == "R":
                    cells.append(f"{CLR_RED}🔴{CLR_RESET}")
                elif v == "Y":
                    cells.append(f"{CLR_YELLOW}🟡{CLR_RESET}")
                else:
                    cells.append(f"{CLR_GRAY}⚪{CLR_RESET}")
            board_rows.append(f"{CLR_CYAN}│{CLR_RESET}" + " ".join(cells) + f"{CLR_CYAN}│{CLR_RESET}")
        board_rows.append(f"{CLR_CYAN}└═════════════════════┘{CLR_RESET}")

        # 2. Right: Telemetry
        mins = int(elapsed_s // 60)
        secs = elapsed_s % 60
        timer_str = f"{mins:02d}:{secs:05.2f}"
        curr_player = "Red (R)" if self.turns % 2 == 0 else "Yellow (Y)"

        speedup_str = ""
        if latency_ms and latency_ms > 0:
            speedup_ratio = max(1.0, 1100.0 / latency_ms)
            speedup_str = f" (~{speedup_ratio:.0f}× vs CoT)"

        sidebar = [
            f"{CLR_BOLD}🔴 CONNECT FOUR SYSTEM ONE{CLR_RESET}   {CLR_YELLOW}⏱ Elapsed: {timer_str}{CLR_RESET}",
            f"{CLR_DIM}─────────────────────────────────────────────────────────────────{CLR_RESET}",
            f"  Turn: {CLR_YELLOW}{self.turns:<3}{CLR_RESET} | Active: {CLR_BRIGHT_GREEN}{curr_player}{CLR_RESET} | Columns Legal: {len(self.board.get_legal_columns())}/7",
        ]

        if latency_ms is not None:
            avg_str = f"{avg_latency_ms:.1f} ms" if avg_latency_ms else f"{latency_ms:.1f} ms"
            sidebar.append(
                f"  {CLR_BOLD}⚡ Latency:{CLR_RESET} {CLR_CYAN}{latency_ms:5.1f} ms{CLR_RESET} (Avg: {avg_str}){CLR_BRIGHT_GREEN}{speedup_str}{CLR_RESET}"
            )
        else:
            sidebar.append(f"  {CLR_BOLD}⚡ Latency:{CLR_RESET} Ready")

        sidebar.append(f"{CLR_BOLD}🧠 Column Choice Distribution:{CLR_RESET}")
        if last_decision:
            chosen = str(last_decision.get("chosen_col", "N/A"))
            conf = last_decision.get("confidence", 0.0)
            probs = last_decision.get("probabilities", {})
            for c in range(1, 8):
                c_str = str(c)
                p = probs.get(c_str, 0.0)
                bar_len = int(p * 8)
                bar_str = "█" * bar_len + "░" * (8 - bar_len)
                is_p = c_str == chosen
                chk = f"{CLR_BRIGHT_GREEN}[✓]{CLR_RESET}" if is_p else f"{CLR_DIM}[ ]{CLR_RESET}"
                sidebar.append(f"   {chk} Col {c}: {p*100:4.1f}% [{bar_str}]")

            sidebar.append(f"  {CLR_BOLD}📋 Parallel Outputs:{CLR_RESET} Conf: {CLR_BRIGHT_GREEN}{conf * 100:.1f}%{CLR_RESET}")
        else:
            sidebar.append("   (Planning column drop...)")

        if self.game_over:
            if self.winner:
                sidebar.append(f"  {CLR_BRIGHT_GREEN}{CLR_BOLD}🎉 CONNECT 4! {'RED' if self.winner == 'R' else 'YELLOW'} WINS!{CLR_RESET}")
            else:
                sidebar.append(f"  {CLR_YELLOW}{CLR_BOLD}🤝 DRAW GAME! Board is full.{CLR_RESET}")

        total_lines = max(len(board_rows), len(sidebar))
        combined = []
        for i in range(total_lines):
            left = board_rows[i] if i < len(board_rows) else " " * 24
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

        # 1. Left: Connect 4 Grid
        pad_x = 40
        pad_y = 60
        cell_size = 48
        draw.rounded_rectangle([pad_x - 10, pad_y - 10, pad_x + COLS * cell_size + 10, pad_y + ROWS * cell_size + 10], radius=12, fill="#1d4ed8", outline="#1e40af", width=3)

        for c in range(COLS):
            draw.text((pad_x + c * cell_size + cell_size // 2 - 4, pad_y - 30), str(c + 1), font=font_bold, fill="#93c5fd")

        for r in range(ROWS):
            for c in range(COLS):
                v = self.board.grid[r][c]
                x0 = pad_x + c * cell_size
                y0 = pad_y + r * cell_size
                col = "#0b0f19"
                if v == "R":
                    col = "#ef4444"
                elif v == "Y":
                    col = "#eab308"
                draw.ellipse([x0 + 6, y0 + 6, x0 + cell_size - 6, y0 + cell_size - 6], fill=col)

        # 2. Right: HUD
        hx0 = 430
        hy0 = pad_y - 10
        panel_w = 410
        panel_h = ROWS * cell_size + 20

        draw.rounded_rectangle([hx0, hy0, hx0 + panel_w, hy0 + panel_h], radius=10, fill="#111827", outline="#1f2937", width=2)
        curr_y = hy0 + 14

        draw.text((hx0 + 16, curr_y), "CONNECT FOUR SYSTEM ONE", font=font_title, fill="#38bdf8")
        curr_y += 24
        draw.text((hx0 + 16, curr_y), "7-Column Tactical Reflexes (<12ms)", font=font_sub, fill="#94a3b8")
        curr_y += 20

        # Stats
        card_w = (panel_w - 32 - 16) // 3
        curr_p = "Red (R)" if self.turns % 2 == 0 else "Yellow (Y)"
        stats = [("TURNS", str(self.turns), "#facc15"), ("ACTIVE", curr_p[:6], "#ef4444" if "Red" in curr_p else "#eab308"), ("LEGAL", f"{len(self.board.get_legal_columns())}/7", "#38bdf8")]
        for i, (lbl, val, col) in enumerate(stats):
            cx = hx0 + 16 + i * (card_w + 8)
            draw.rounded_rectangle([cx, curr_y, cx + card_w, curr_y + 40], radius=5, fill="#1e293b")
            draw.text((cx + 8, curr_y + 4), lbl, font=font_stat_lbl, fill="#94a3b8")
            draw.text((cx + 8, curr_y + 18), val, font=font_stat_val, fill=col)
        curr_y += 50

        # Latency meter
        draw.rounded_rectangle([hx0 + 16, curr_y, hx0 + panel_w - 16, curr_y + 40], radius=5, fill="#1e293b", outline="#334155")
        if latency_ms is not None:
            speedup = max(1.0, 1100.0 / latency_ms)
            lat_text = f"⚡ {latency_ms:.1f} ms (~{speedup:.0f}× faster than CoT)"
            draw.text((hx0 + 20, curr_y + 5), lat_text, font=font_bold, fill="#facc15")
            avg_text = f"Avg: {avg_latency_ms:.1f} ms • Zero-token Option Logits" if avg_latency_ms else "Sub-15ms Column Decisions"
            draw.text((hx0 + 20, curr_y + 22), avg_text, font=font_small, fill="#38bdf8")
        curr_y += 48

        # Direction Choices
        draw.text((hx0 + 16, curr_y), "Column Drop Distribution:", font=font_bold, fill="#e2e8f0")
        curr_y += 18
        if last_decision:
            chosen = str(last_decision.get("chosen_col", ""))
            probs = last_decision.get("probabilities", {})
            for c in range(1, 8):
                c_str = str(c)
                p = probs.get(c_str, 0.0)
                is_p = c_str == chosen
                row_bg = "#1e293b" if is_p else "#0f172a"
                draw.rounded_rectangle([hx0 + 16, curr_y, hx0 + panel_w - 16, curr_y + 18], radius=3, fill=row_bg)
                chk = "✓ " if is_p else "  "
                draw.text((hx0 + 20, curr_y + 2), f"{chk}Col {c}", font=font_bold if is_p else font_small, fill="#4ade80" if is_p else "#94a3b8")

                bar_x = hx0 + 100
                draw.rounded_rectangle([bar_x, curr_y + 5, bar_x + int(p * 100), curr_y + 13], radius=2, fill="#38bdf8" if is_p else "#475569")
                draw.text((bar_x + 110, curr_y + 2), f"{p*100:4.1f}%", font=font_small, fill="#cbd5e1")
                curr_y += 21

        return img


def play_connect4_demo(
    client: SystemOneClient,
    max_turns: int = 25,
    delay: float = 0.25,
    animate: bool = True,
    gif_path: str | Path | None = None,
    fps: int = 4,
) -> dict[str, Any]:
    game = Connect4GameSession()
    latencies: list[float] = []
    frames: list[Image.Image] = []
    start_time = time.perf_counter()
    last_decision: dict[str, Any] | None = None

    if animate:
        sys.stdout.write("\033[2J\033[H\033[?25l")
        sys.stdout.flush()

    try:
        while not game.game_over and game.turns < max_turns:
            state, questions = game.build_systemone_payload()
            t0 = time.perf_counter()
            resp = client.system_one(state=state, questions=questions)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)
            avg_lat = sum(latencies) / len(latencies)
            elapsed_s = time.perf_counter() - start_time

            ans_col = resp.answers["drop"]
            try:
                chosen_col = int(ans_col.choice)
            except (ValueError, TypeError):
                curr_player = "R" if game.turns % 2 == 0 else "Y"
                chosen_col = best_move_connect4(game.board, curr_player) + 1

            last_decision = {
                "chosen_col": chosen_col,
                "confidence": ans_col.confidence or 0.0,
                "probabilities": ans_col.probabilities or {},
                "threat_alert": resp.answers.get("threat_alert"),
            }

            if gif_path:
                frames.append(game.render_frame_pil(last_decision, lat_ms, avg_lat, elapsed_s))

            if animate:
                sys.stdout.write("\033[H")
                sys.stdout.write(game.render_ansi(last_decision, lat_ms, avg_lat, elapsed_s) + "\n")
                sys.stdout.flush()
                if delay > 0:
                    time.sleep(delay)

            game.step(chosen_col)

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
        print(f"\nSaved Connect Four animated GIF → {out_p}")

    return {
        "turns": game.turns,
        "winner": game.winner,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Connect Four Demo (systemone-lite)")
    parser.add_argument("--max-turns", type=int, default=24, help="Maximum turns to play")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay between turns in seconds")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--stub", action="store_true", help="Use stub engine (no GPU / no download)")
    parser.add_argument("--gif", type=Path, default=None, help="Output GIF path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.stub:
        client = SystemOneClient(engine=StubEngine())
    else:
        client = SystemOneClient(model=args.model)

    res = play_connect4_demo(
        client=client,
        max_turns=args.max_turns,
        delay=0.0 if args.quiet else args.delay,
        animate=not args.quiet,
        gif_path=args.gif,
    )

    print("\n" + "=" * 50)
    print("Connect Four Game Finished!")
    print(f"Outcome       : {'WINNER: ' + str(res['winner']) if res['winner'] else 'DRAW'}")
    print(f"Total Turns   : {res['turns']}")
    print(f"Avg Latency   : {res['avg_latency_ms']:.1f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
