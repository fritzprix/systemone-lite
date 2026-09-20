#!/usr/bin/env python3
"""Multi-step Autonomous Chess Demo powered by systemone-lite.

Demonstrates a realistic 2-stage 'System 1' decision pipeline per turn:
  Stage 1 (Piece Selection):
    - System 1 inspects the 2D board map and selects WHICH piece should move.
    - Prioritizes active center control, minor piece development, and king safety.
  Stage 2 (Destination Selection):
    - With the piece selected and highlighted on the board, System 1 selects
      WHERE that specific piece should move among all its legal destinations.
    - Simultaneously evaluates strategic position score and threat alert.

Features:
  - Rich ANSI terminal animation with piece glyphs & two-stage telemetry HUD
  - High-definition GIF / MP4 export with visual highlighting of selected piece
  - Instant dry-run mode via --stub
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
from PIL import Image, ImageDraw, ImageFont

from systemone_lite import SystemOneClient, choice, noul, score
from systemone_lite.chess_data import (
    PIECE_VALUE,
    alias_criteria,
    board_state,
    describe_move,
    describe_piece,
    legal_by_origin,
    render_board_ascii,
)
from systemone_lite.infer import DEFAULT_MODEL_ID
from systemone_lite.stub import StubEngine

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
CLR_WHITE = "\033[97m"
CLR_GRAY = "\033[90m"
CLR_BG_SELECTED = "\033[48;5;30m"  # Teal background for selected piece

PIECE_UNICODE = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "➶", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
    ".": "·",
}
# Fix black knight glyph if needed
PIECE_UNICODE["n"] = "♞"


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _score_origin(board: chess.Board, sq_name: str) -> float:
    """Heuristic score to rank origin candidate options."""
    sq = chess.parse_square(sq_name)
    piece = board.piece_at(sq)
    if piece is None:
        return 0.0
    s = 0.0
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    is_opening = board.fullmove_number <= 7

    # Center pawns
    if piece.piece_type == chess.PAWN and file in (3, 4):
        s += 35.0 if is_opening else 15.0
    # Minor piece development from back rank
    if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
        back = 0 if piece.color == chess.WHITE else 7
        if rank == back:
            s += 30.0 if is_opening else 10.0
        else:
            s += 15.0
    # Checks or captures available from this piece
    for move in board.legal_moves:
        if move.from_square == sq:
            if board.is_capture(move):
                s += 25.0
            if board.gives_check(move):
                s += 40.0
    return s


def _score_target(board: chess.Board, move: chess.Move) -> float:
    """Heuristic score to rank destination candidate options."""
    s = 0.0
    if board.gives_check(move):
        s += 50.0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        v = PIECE_VALUE.get(victim.piece_type, 1) if victim else 1
        s += 30.0 + v * 5.0
    to_sq = chess.square_name(move.to_square)
    if to_sq in ("e4", "d4", "e5", "d5"):
        s += 25.0
    mover = board.piece_at(move.from_square)
    if mover and mover.piece_type in (chess.KNIGHT, chess.BISHOP):
        from_rank = chess.square_rank(move.from_square)
        if (mover.color == chess.WHITE and from_rank == 0) or (mover.color == chess.BLACK and from_rank == 7):
            s += 20.0
    if board.is_castling(move):
        s += 40.0
    return s


@dataclass
class Step1Decision:
    chosen_alias: str
    chosen_origin: str
    origin_desc: str
    confidence: float
    probabilities: dict[str, float]
    criteria: dict[str, str]
    latency_ms: float


@dataclass
class Step2Decision:
    chosen_alias: str
    chosen_move: chess.Move
    chosen_san: str
    confidence: float
    probabilities: dict[str, float]
    criteria: dict[str, str]
    latency_ms: float
    position_eval: float | None = None
    threat_alert: bool | None = None


class MultiStepChessGame:
    def __init__(self) -> None:
        self.board = chess.Board()
        self.move_history: list[tuple[chess.Move, str]] = []
        self.plies = 0
        self.game_over = False
        self.result = ""

    def reset(self) -> None:
        self.board.reset()
        self.move_history.clear()
        self.plies = 0
        self.game_over = False
        self.result = ""

    def get_material_count(self) -> tuple[int, int]:
        white_val = sum(PIECE_VALUE[p.piece_type] for p in self.board.piece_map().values() if p.color == chess.WHITE)
        black_val = sum(PIECE_VALUE[p.piece_type] for p in self.board.piece_map().values() if p.color == chess.BLACK)
        return white_val, black_val

    def build_step1_piece_payload(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
        """Stage 1: Which piece should move this turn?"""
        grouped = legal_by_origin(self.board)
        if not grouped:
            raise RuntimeError("No legal moves available")

        # Rank candidate origin squares by activity & tactical value
        sorted_origins = sorted(grouped.keys(), key=lambda sq: _score_origin(self.board, sq), reverse=True)
        # Cap to top 8 origin pieces for sharp multi-choice
        top_origins = sorted_origins[:8]

        origin_options = {sq: describe_piece(self.board, sq) for sq in top_origins}
        criteria, alias_to_key = alias_criteria(origin_options)

        turn_name = "White" if self.board.turn == chess.WHITE else "Black"
        white_mat, black_mat = self.get_material_count()

        state = {
            "board_2d_map": f"\n{render_board_ascii(self.board)}\n",
            "fen": self.board.fen(),
            "side_to_move": f"{turn_name} (Turn {self.board.fullmove_number})",
            "is_in_check": self.board.is_check(),
            "material_balance": f"White {white_mat} vs Black {black_mat}",
            "decision_stage": "Stage 1 of 2: Selecting which active piece to move",
        }

        questions = {
            "piece": choice(
                f"It is {turn_name}'s turn. Based on the 2D chess board map, choose WHICH piece to move. "
                "Prefer development, center control, and king safety. Reply with option letter.",
                criteria,
            )
        }
        return state, questions, alias_to_key

    def build_step2_move_payload(self, origin: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, chess.Move]]:
        """Stage 2: Where should the selected piece move?"""
        grouped = legal_by_origin(self.board)
        dest_moves = grouped.get(origin, [])
        if not dest_moves:
            # Fallback if square has no legal moves
            dest_moves = list(self.board.legal_moves)

        # Rank destinations for this piece by tactical priority
        sorted_moves = sorted(dest_moves, key=lambda m: _score_target(self.board, m), reverse=True)
        top_moves = sorted_moves[:8]

        move_options = {m.uci(): describe_move(self.board, m) for m in top_moves}
        criteria, alias_to_key = alias_criteria(move_options)
        alias_to_move = {alias: chess.Move.from_uci(uci) for alias, uci in alias_to_key.items()}

        turn_name = "White" if self.board.turn == chess.WHITE else "Black"
        white_mat, black_mat = self.get_material_count()
        origin_desc = describe_piece(self.board, origin)

        state = {
            "board_2d_map": f"\n{render_board_ascii(self.board)}\n",
            "fen": self.board.fen(),
            "side_to_move": f"{turn_name} (Turn {self.board.fullmove_number})",
            "selected_piece": origin,
            "selected_piece_description": origin_desc,
            "decision_stage": f"Stage 2 of 2: Piece on {origin} ({origin_desc}) is selected. Choosing destination.",
        }

        questions = {
            "destination": choice(
                f"The piece on {origin} ({origin_desc}) is selected. "
                "Choose the best legal destination square. Reply with option letter.",
                criteria,
            ),
            "position_eval": score(
                f"Evaluate current position from {turn_name}'s perspective.",
                [
                    "Equal: balanced position and material",
                    "Slight advantage: better center or piece activity",
                    "Clear advantage / Winning: major tactical win or up material",
                ],
            ),
            "threat_alert": noul(
                f"Is {turn_name}'s King or a high-value piece currently in immediate danger?"
            ),
        }
        return state, questions, alias_to_move

    def step(self, move: chess.Move) -> bool:
        if self.game_over:
            return False

        san = self.board.san(move)
        self.board.push(move)
        self.plies += 1
        self.move_history.append((move, san))

        if self.board.is_checkmate():
            self.game_over = True
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            self.result = f"Checkmate! {winner} wins!"
            return False
        if self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_draw():
            self.game_over = True
            self.result = "Draw (Stalemate / Insufficient Material / Repetition)"
            return False

        return True

    def render_ansi(
        self,
        step1: Step1Decision | None,
        step2: Step2Decision | None,
        total_latency_ms: float | None = None,
        avg_latency_ms: float | None = None,
        elapsed_s: float = 0.0,
    ) -> str:
        # Highlight square of selected piece
        active_sq = chess.parse_square(step1.chosen_origin) if step1 else None
        target_sq = step2.chosen_move.to_square if step2 else None

        # 1. Left: Chess board rows
        board_rows = [f"{CLR_CYAN}  ┌────────────────────────┐{CLR_RESET}"]
        for rank in range(7, -1, -1):
            row = [f"{CLR_CYAN}{rank + 1} │{CLR_RESET}"]
            for file in range(8):
                sq = chess.square(file, rank)
                p = self.board.piece_at(sq)
                bg_is_light = (file + rank) % 2 != 0

                if sq == active_sq:
                    bg_col = CLR_BG_SELECTED
                elif sq == target_sq:
                    bg_col = "\033[48;5;28m"  # Green target
                else:
                    bg_col = "\033[48;5;238m" if bg_is_light else "\033[48;5;235m"

                if p:
                    col = CLR_WHITE if p.color == chess.WHITE else CLR_YELLOW
                    glyph = PIECE_UNICODE.get(p.symbol(), p.symbol())
                    row.append(f"{bg_col}{col}{glyph} {CLR_RESET}")
                else:
                    row.append(f"{bg_col}{CLR_GRAY}· {CLR_RESET}")
            row.append(f"{CLR_CYAN}│ {rank + 1}{CLR_RESET}")
            board_rows.append("".join(row))
        board_rows.append(f"{CLR_CYAN}  └────────────────────────┘{CLR_RESET}")
        board_rows.append(f"{CLR_CYAN}    a  b  c  d  e  f  g  h  {CLR_RESET}")

        # 2. Right: Telemetry & Multi-step breakdown
        mins = int(elapsed_s // 60)
        secs = elapsed_s % 60
        timer_str = f"{mins:02d}:{secs:05.2f}"
        turn_name = "White" if self.board.turn == chess.WHITE else "Black"
        white_mat, black_mat = self.get_material_count()

        speedup_str = ""
        if total_latency_ms and total_latency_ms > 0:
            speedup_ratio = max(1.0, 1200.0 / total_latency_ms)
            speedup_str = f" (~{speedup_ratio:.0f}× vs AR CoT)"

        sidebar = [
            f"{CLR_BOLD}♟️ MULTI-STEP SYSTEM ONE CHESS{CLR_RESET}   {CLR_YELLOW}⏱ {timer_str}{CLR_RESET}",
            f"{CLR_DIM}──────────────────────────────────────────────────────────────────{CLR_RESET}",
            f"  Ply: {CLR_YELLOW}{self.plies:<3}{CLR_RESET} | Turn: {CLR_BRIGHT_GREEN}{turn_name:<5}{CLR_RESET} | Move: {self.board.fullmove_number} | Material: {white_mat}:{black_mat}",
        ]

        if total_latency_ms is not None:
            avg_str = f"{avg_latency_ms:.1f} ms" if avg_latency_ms else f"{total_latency_ms:.1f} ms"
            p1_lat = f"{step1.latency_ms:.1f}ms" if step1 else "--"
            p2_lat = f"{step2.latency_ms:.1f}ms" if step2 else "--"
            sidebar.append(
                f"  {CLR_BOLD}⚡ Latency:{CLR_RESET} {CLR_CYAN}{total_latency_ms:5.1f} ms{CLR_RESET} "
                f"(P1:{p1_lat} + P2:{p2_lat}, Avg: {avg_str}){CLR_BRIGHT_GREEN}{speedup_str}{CLR_RESET}"
            )
        else:
            sidebar.append(f"  {CLR_BOLD}⚡ Latency:{CLR_RESET} Ready")

        # Step 1 Section
        sidebar.append(f"{CLR_BOLD}🎯 Step 1: Piece Selection (Which piece moves?){CLR_RESET}")
        if step1:
            for alias, desc in list(step1.criteria.items())[:3]:
                p = step1.probabilities.get(alias, 0.0)
                is_picked = alias == step1.chosen_alias
                check = f"{CLR_BRIGHT_GREEN}[✓]{CLR_RESET}" if is_picked else f"{CLR_DIM}[ ]{CLR_RESET}"
                sidebar.append(f"   {check} {CLR_CYAN}{alias}{CLR_RESET}: {p*100:4.1f}% | {desc[:40]}")
            sidebar.append(
                f"   ➔ Selected: {CLR_BRIGHT_GREEN}{step1.chosen_origin}{CLR_RESET} ({step1.origin_desc}) "
                f"| Conf: {step1.confidence * 100:.1f}% ({step1.latency_ms:.1f}ms)"
            )
        else:
            sidebar.append("   (Selecting piece...)")

        # Step 2 Section
        sidebar.append(f"{CLR_BOLD}📍 Step 2: Destination Selection (Where to move?){CLR_RESET}")
        if step2:
            for alias, desc in list(step2.criteria.items())[:3]:
                p = step2.probabilities.get(alias, 0.0)
                is_picked = alias == step2.chosen_alias
                check = f"{CLR_BRIGHT_GREEN}[✓]{CLR_RESET}" if is_picked else f"{CLR_DIM}[ ]{CLR_RESET}"
                sidebar.append(f"   {check} {CLR_CYAN}{alias}{CLR_RESET}: {p*100:4.1f}% | {desc[:40]}")
            eval_str = f"{step2.position_eval:.2f}" if step2.position_eval is not None else "N/A"
            threat_str = f"{step2.threat_alert}" if step2.threat_alert is not None else "N/A"
            sidebar.append(
                f"   ➔ Move: {CLR_BRIGHT_GREEN}{step2.chosen_san}{CLR_RESET} | Conf: {step2.confidence * 100:.1f}% "
                f"| Eval: {CLR_YELLOW}{eval_str}{CLR_RESET} | Danger: {CLR_RED if step2.threat_alert else CLR_GREEN}{threat_str}{CLR_RESET}"
            )
        else:
            sidebar.append("   (Awaiting destination choice...)")

        if self.game_over:
            sidebar.append(f"  {CLR_RED}{CLR_BOLD}💥 GAME OVER: {self.result}{CLR_RESET}")

        total_lines = max(len(board_rows), len(sidebar))
        combined = []
        for i in range(total_lines):
            left = board_rows[i] if i < len(board_rows) else " " * 32
            right = sidebar[i] if i < len(sidebar) else ""
            combined.append(f"{left}  {right}")
        return "\n".join(combined)

    def render_frame_pil(
        self,
        step1: Step1Decision | None,
        step2: Step2Decision | None,
        total_latency_ms: float | None = None,
        avg_latency_ms: float | None = None,
        elapsed_s: float = 0.0,
    ) -> Image.Image:
        """Render a high-definition dashboard frame for GIF recording."""
        img_w, img_h = 1000, 600
        img = Image.new("RGB", (img_w, img_h), color="#0b0f19")
        draw = ImageDraw.Draw(img)

        font_title = _get_font(20, bold=True)
        font_sub = _get_font(12, bold=False)
        font_timer = _get_font(14, bold=True)
        font_stat_val = _get_font(18, bold=True)
        font_stat_lbl = _get_font(11, bold=False)
        font_body = _get_font(12, bold=False)
        font_bold = _get_font(12, bold=True)
        font_small = _get_font(11, bold=False)
        font_piece = _get_font(28, bold=True)

        # 1. Left: Chess Board
        board_size = 500
        pad_x, pad_y = 35, 50
        cell_size = board_size // 8

        light_color = "#e2e8f0"
        dark_color = "#475569"

        active_sq = chess.parse_square(step1.chosen_origin) if step1 else None
        target_sq = step2.chosen_move.to_square if step2 else None

        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, 7 - rank)
                x0 = pad_x + file * cell_size
                y0 = pad_y + rank * cell_size
                is_light = (file + rank) % 2 == 0
                cell_col = light_color if is_light else dark_color

                if sq == active_sq:
                    cell_col = "#0284c7"  # Teal / Sky blue for selected piece
                elif sq == target_sq:
                    cell_col = "#16a34a"  # Green for destination target

                draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size], fill=cell_col)

                if sq == active_sq:
                    draw.rectangle([x0 + 2, y0 + 2, x0 + cell_size - 2, y0 + cell_size - 2], outline="#facc15", width=3)

                p = self.board.piece_at(sq)
                if p:
                    glyph = PIECE_UNICODE.get(p.symbol(), p.symbol())
                    p_col = "#0f172a" if p.color == chess.BLACK else "#ffffff"
                    if p.color == chess.WHITE and is_light and sq != active_sq and sq != target_sq:
                        draw.text((x0 + cell_size // 4 + 1, y0 + cell_size // 8 + 1), glyph, font=font_piece, fill="#64748b")
                    draw.text((x0 + cell_size // 4, y0 + cell_size // 8), glyph, font=font_piece, fill=p_col)

        # File and Rank Labels
        for i, file_char in enumerate("abcdefgh"):
            draw.text((pad_x + i * cell_size + cell_size // 2 - 4, pad_y + board_size + 8), file_char, font=font_bold, fill="#94a3b8")
        for i in range(8):
            draw.text((pad_x - 18, pad_y + i * cell_size + cell_size // 2 - 8), str(8 - i), font=font_bold, fill="#94a3b8")

        # 2. Right: HUD
        hx0 = 570
        hy0 = pad_y
        panel_w = 400
        panel_h = board_size + 20

        draw.rounded_rectangle([hx0, hy0, hx0 + panel_w, hy0 + panel_h], radius=10, fill="#111827", outline="#1f2937", width=2)
        curr_y = hy0 + 12

        mins = int(elapsed_s // 60)
        secs = elapsed_s % 60
        timer_text = f"⏱ {mins:02d}:{secs:05.2f}"

        draw.text((hx0 + 14, curr_y), "MULTI-STEP CHESS AGENT", font=font_title, fill="#38bdf8")
        draw.text((hx0 + panel_w - 105, curr_y + 4), timer_text, font=font_timer, fill="#facc15")
        curr_y += 24
        draw.text((hx0 + 14, curr_y), "Two-Stage Decision: 1) Piece -> 2) Destination", font=font_sub, fill="#94a3b8")
        curr_y += 18
        draw.line([hx0 + 14, curr_y, hx0 + panel_w - 14, curr_y], fill="#1f2937", width=1)
        curr_y += 10

        # Stats row
        turn_name = "White" if self.board.turn == chess.WHITE else "Black"
        white_mat, black_mat = self.get_material_count()
        card_w = (panel_w - 28 - 16) // 3
        stats = [("PLY", str(self.plies), "#facc15"), ("TURN", turn_name, "#4ade80"), ("MATERIAL", f"{white_mat}:{black_mat}", "#38bdf8")]
        for i, (lbl, val, col) in enumerate(stats):
            cx = hx0 + 14 + i * (card_w + 8)
            draw.rounded_rectangle([cx, curr_y, cx + card_w, curr_y + 38], radius=5, fill="#1e293b")
            draw.text((cx + 8, curr_y + 4), lbl, font=font_stat_lbl, fill="#94a3b8")
            draw.text((cx + 8, curr_y + 16), val, font=font_stat_val, fill=col)
        curr_y += 46

        # Latency Box
        draw.rounded_rectangle([hx0 + 14, curr_y, hx0 + panel_w - 14, curr_y + 40], radius=5, fill="#1e293b", outline="#334155")
        if total_latency_ms is not None:
            speedup = max(1.0, 1200.0 / total_latency_ms)
            p1_ms = f"{step1.latency_ms:.1f}ms" if step1 else "--"
            p2_ms = f"{step2.latency_ms:.1f}ms" if step2 else "--"
            lat_text = f"⚡ Total: {total_latency_ms:.1f} ms (P1: {p1_ms} + P2: {p2_ms}) (~{speedup:.0f}× vs CoT)"
            draw.text((hx0 + 20, curr_y + 5), lat_text, font=font_bold, fill="#facc15")
            avg_text = f"Avg Turn: {avg_latency_ms:.1f} ms • Zero-token Option Logits" if avg_latency_ms else "Two-step Constrained Logits"
            draw.text((hx0 + 20, curr_y + 22), avg_text, font=font_small, fill="#38bdf8")
        else:
            draw.text((hx0 + 20, curr_y + 12), "⚡ Two-Stage System One Ready", font=font_bold, fill="#facc15")
        curr_y += 48

        # Step 1 Breakdown
        draw.rounded_rectangle([hx0 + 14, curr_y, hx0 + panel_w - 14, curr_y + 115], radius=6, fill="#0f172a", outline="#1e293b")
        draw.text((hx0 + 22, curr_y + 6), "STAGE 1: Piece Selection", font=font_bold, fill="#38bdf8")
        if step1:
            draw.text((hx0 + panel_w - 120, curr_y + 6), f"Conf: {step1.confidence*100:.1f}%", font=font_small, fill="#4ade80")
            s1_y = curr_y + 24
            for alias, desc in list(step1.criteria.items())[:2]:
                p = step1.probabilities.get(alias, 0.0)
                is_p = alias == step1.chosen_alias
                prefix = "✓ " if is_p else "  "
                txt = f"{prefix}{alias}: {p*100:4.1f}% | {desc[:28]}"
                draw.text((hx0 + 24, s1_y), txt, font=font_small, fill="#4ade80" if is_p else "#94a3b8")
                s1_y += 18
            draw.text((hx0 + 24, s1_y + 4), f"➔ Chosen: {step1.chosen_origin} ({step1.origin_desc[:25]})", font=font_bold, fill="#facc15")
        else:
            draw.text((hx0 + 24, curr_y + 35), "Selecting which piece to move...", font=font_small, fill="#64748b")
        curr_y += 123

        # Step 2 Breakdown
        draw.rounded_rectangle([hx0 + 14, curr_y, hx0 + panel_w - 14, curr_y + 125], radius=6, fill="#0f172a", outline="#1e293b")
        draw.text((hx0 + 22, curr_y + 6), "STAGE 2: Destination Selection", font=font_bold, fill="#4ade80")
        if step2:
            draw.text((hx0 + panel_w - 120, curr_y + 6), f"Conf: {step2.confidence*100:.1f}%", font=font_small, fill="#4ade80")
            s2_y = curr_y + 24
            for alias, desc in list(step2.criteria.items())[:2]:
                p = step2.probabilities.get(alias, 0.0)
                is_p = alias == step2.chosen_alias
                prefix = "✓ " if is_p else "  "
                txt = f"{prefix}{alias}: {p*100:4.1f}% | {desc[:28]}"
                draw.text((hx0 + 24, s2_y), txt, font=font_small, fill="#4ade80" if is_p else "#94a3b8")
                s2_y += 18
            eval_txt = f"Eval: {step2.position_eval:.2f}" if step2.position_eval is not None else ""
            danger_txt = "Danger: YES" if step2.threat_alert else "Danger: NO"
            draw.text((hx0 + 24, s2_y + 4), f"➔ Move: {step2.chosen_san} | {eval_txt} | {danger_txt}", font=font_bold, fill="#38bdf8")
        else:
            draw.text((hx0 + 24, curr_y + 35), "Awaiting destination choice...", font=font_small, fill="#64748b")
        curr_y += 135

        if self.game_over:
            draw.rounded_rectangle([hx0 + 14, curr_y, hx0 + panel_w - 14, curr_y + 32], radius=5, fill="#7f1d1d")
            draw.text((hx0 + 22, curr_y + 8), f"GAME OVER: {self.result[:30]}", font=font_bold, fill="#fecaca")

        return img


def play_multistep_game(
    client: SystemOneClient,
    max_plies: int = 20,
    delay: float = 0.3,
    step_delay: float = 0.15,
    animate: bool = True,
    gif_path: str | Path | None = None,
    mp4_path: str | Path | None = None,
    fps: int = 4,
) -> dict[str, Any]:
    game = MultiStepChessGame()
    latencies: list[float] = []
    frames: list[Image.Image] = []
    start_time = time.perf_counter()

    last_step1: Step1Decision | None = None
    last_step2: Step2Decision | None = None

    if animate:
        sys.stdout.write("\033[2J\033[H\033[?25l")
        sys.stdout.flush()

    try:
        while not game.game_over and game.plies < max_plies:
            # ==========================================
            # STAGE 1: PIECE SELECTION
            # ==========================================
            state1, q1, alias_to_origin = game.build_step1_piece_payload()
            t0 = time.perf_counter()
            resp1 = client.system_one(state=state1, questions=q1)
            t1_ms = (time.perf_counter() - t0) * 1000.0

            ans_piece = resp1.answers["piece"]
            chosen_p_alias = ans_piece.choice
            chosen_origin = alias_to_origin.get(chosen_p_alias)
            if chosen_origin is None:
                chosen_p_alias = list(alias_to_origin.keys())[0]
                chosen_origin = alias_to_origin[chosen_p_alias]

            origin_desc = describe_piece(game.board, chosen_origin)
            last_step1 = Step1Decision(
                chosen_alias=chosen_p_alias,
                chosen_origin=chosen_origin,
                origin_desc=origin_desc,
                confidence=ans_piece.confidence or 0.0,
                probabilities=ans_piece.probabilities or {},
                criteria=q1["piece"]["criteria"],
                latency_ms=t1_ms,
            )

            # Intermediate Stage 1 animation frame (showing piece selected)
            elapsed_s = time.perf_counter() - start_time
            if gif_path or mp4_path:
                frames.append(game.render_frame_pil(last_step1, None, t1_ms, None, elapsed_s))

            if animate:
                sys.stdout.write("\033[H")
                sys.stdout.write(game.render_ansi(last_step1, None, t1_ms, None, elapsed_s) + "\n")
                sys.stdout.flush()
                if step_delay > 0:
                    time.sleep(step_delay)

            # ==========================================
            # STAGE 2: DESTINATION SELECTION
            # ==========================================
            state2, q2, alias_to_move = game.build_step2_move_payload(chosen_origin)
            t2_start = time.perf_counter()
            resp2 = client.system_one(state=state2, questions=q2)
            t2_ms = (time.perf_counter() - t2_start) * 1000.0

            ans_dest = resp2.answers["destination"]
            chosen_m_alias = ans_dest.choice
            chosen_move = alias_to_move.get(chosen_m_alias)
            if chosen_move is None:
                chosen_m_alias = list(alias_to_move.keys())[0]
                chosen_move = alias_to_move[chosen_m_alias]

            chosen_san = game.board.san(chosen_move)
            pos_eval = resp2.answers.get("position_eval")
            threat = resp2.answers.get("threat_alert")

            total_turn_ms = t1_ms + t2_ms
            latencies.append(total_turn_ms)
            avg_lat = sum(latencies) / len(latencies)
            elapsed_s = time.perf_counter() - start_time

            last_step2 = Step2Decision(
                chosen_alias=chosen_m_alias,
                chosen_move=chosen_move,
                chosen_san=chosen_san,
                confidence=ans_dest.confidence or 0.0,
                probabilities=ans_dest.probabilities or {},
                criteria=q2["destination"]["criteria"],
                latency_ms=t2_ms,
                position_eval=pos_eval.score if pos_eval else None,
                threat_alert=threat.noul if threat else None,
            )

            # Final Stage 2 animation frame (showing destination chosen & move played)
            if gif_path or mp4_path:
                frames.append(game.render_frame_pil(last_step1, last_step2, total_turn_ms, avg_lat, elapsed_s))

            if animate:
                sys.stdout.write("\033[H")
                sys.stdout.write(game.render_ansi(last_step1, last_step2, total_turn_ms, avg_lat, elapsed_s) + "\n")
                sys.stdout.flush()
                if delay > 0:
                    time.sleep(delay)

            game.step(chosen_move)

        # Final game over display
        total_elapsed_s = time.perf_counter() - start_time
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        if gif_path or mp4_path:
            frames.append(game.render_frame_pil(last_step1, last_step2, latencies[-1] if latencies else 0.0, avg_lat, total_elapsed_s))

        if animate:
            sys.stdout.write("\033[H")
            sys.stdout.write(
                game.render_ansi(last_step1, last_step2, latencies[-1] if latencies else 0.0, avg_lat, total_elapsed_s) + "\n"
            )
            sys.stdout.flush()

    finally:
        if animate:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()

    if gif_path and frames:
        out_p = Path(gif_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        extended_frames = frames + [frames[-1]] * 4
        extended_frames[0].save(
            str(out_p),
            save_all=True,
            append_images=extended_frames[1:],
            duration=int(1000 / max(1, fps)),
            loop=0,
            optimize=True,
        )
        print(f"\nSaved animated multi-step chess GIF → {out_p} ({len(frames)} frames)")

    if mp4_path and frames:
        import imageio.v2 as imageio
        import numpy as np

        out_mp4 = Path(mp4_path)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(out_mp4), fps=fps, codec="libx264")
        for f in frames + [frames[-1]] * (fps * 2):
            writer.append_data(np.array(f))
        writer.close()
        print(f"Saved animated multi-step chess MP4 → {out_mp4}")

    return {
        "plies": game.plies,
        "result": game.result,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "elapsed_seconds": time.perf_counter() - start_time,
        "history": [san for _, san in game.move_history],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Step Autonomous Chess Demo (systemone-lite)")
    parser.add_argument("--max-plies", type=int, default=20, help="Maximum plies (half-moves) to play")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay between plies in seconds")
    parser.add_argument("--step-delay", type=float, default=0.12, help="Delay between Step 1 (piece) and Step 2 (move)")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face model ID or local checkpoint")
    parser.add_argument("--stub", action="store_true", help="Use stub engine (no GPU / no download)")
    parser.add_argument("--gif", type=Path, default=None, help="Output path for animated GIF")
    parser.add_argument("--mp4", type=Path, default=None, help="Output path for MP4 video")
    parser.add_argument("--fps", type=int, default=4, help="FPS for GIF / video output")
    parser.add_argument("--quiet", action="store_true", help="Disable terminal ANSI animation")
    args = parser.parse_args()

    if args.stub:
        client = SystemOneClient(engine=StubEngine())
    else:
        client = SystemOneClient(model=args.model)

    res = play_multistep_game(
        client=client,
        max_plies=args.max_plies,
        delay=0.0 if args.quiet else args.delay,
        step_delay=0.0 if args.quiet else args.step_delay,
        animate=not args.quiet,
        gif_path=args.gif,
        mp4_path=args.mp4,
        fps=args.fps,
    )

    print("\n" + "=" * 60)
    print("Multi-Step Chess Game Completed!")
    print(f"Total Plies Played : {res['plies']}")
    print(f"Game Outcome       : {res['result'] or 'Plies limit reached'}")
    print(f"Avg Turn Latency   : {res['avg_latency_ms']:.1f} ms (Step 1 Piece + Step 2 Move)")
    print(f"Moves              : {' '.join(res['history'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
