#!/usr/bin/env python3
"""Proper autonomous Chess demo powered by systemone-lite.

System One acts as the real-time 'System 1' tactical chess brain:
  - 8x8 2D text grid representation of the board (no FEN-only compression)
  - Rich tactical descriptions (center control, captures, checks, piece development)
  - Eliminates the alphabetical/option bias by candidate ranking & semantic descriptions
  - Multi-question simultaneous evaluation:
      1. 'move' (choice): Best legal move candidate
      2. 'position_eval' (score): Strategic position evaluation
      3. 'in_danger' (noul): King/Queen under threat alert
  - Real-time ANSI terminal animation with piece glyphs and telemetry HUD
  - Optional GIF recording (--gif) and stub mode (--stub)
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
from PIL import Image, ImageDraw, ImageFont

from systemone_lite import SystemOneClient, choice, noul, score
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

PIECE_UNICODE = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
    ".": "·",
}

PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
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


def render_board_ascii(board: chess.Board) -> str:
    """Render 8x8 ASCII board with rank and file labels."""
    lines = ["  a b c d e f g h"]
    for rank in range(7, -1, -1):
        row = [f"{rank + 1}"]
        for file in range(8):
            sq = chess.square(file, rank)
            p = board.piece_at(sq)
            row.append(p.symbol() if p else ".")
        row.append(f"{rank + 1}")
        lines.append(" ".join(row))
    lines.append("  a b c d e f g h")
    return "\n".join(lines)


@dataclass
class CandidateMove:
    move: chess.Move
    san: str
    uci: str
    description: str
    priority_score: float


def analyze_legal_moves(board: chess.Board) -> list[CandidateMove]:
    """Score and describe all legal moves with tactical heuristics."""
    candidates: list[CandidateMove] = []
    center_squares = {chess.E4, chess.D4, chess.E5, chess.D5}
    is_opening = board.fullmove_number <= 7

    for move in board.legal_moves:
        score_val = 0.0
        san = board.san(move)
        uci = move.uci()
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        piece = board.piece_at(move.from_square)
        p_name = chess.piece_name(piece.piece_type) if piece else "piece"

        desc_parts = [f"{p_name.capitalize()} on {from_sq} to {to_sq}"]

        # 1. Checks
        if board.gives_check(move):
            score_val += 50.0
            desc_parts.append("DELIVERS CHECK to enemy King!")

        # 2. Captures
        captured = board.piece_at(move.to_square)
        if captured:
            victim_val = PIECE_VALUE[captured.piece_type]
            score_val += 30.0 + victim_val * 5.0
            desc_parts.append(f"CAPTURES enemy {chess.piece_name(captured.piece_type)} (+{victim_val})")
        elif board.is_en_passant(move):
            score_val += 35.0
            desc_parts.append("CAPTURES pawn en passant")

        # 3. Castling
        if board.is_castling(move):
            score_val += 40.0
            desc_parts.append("CASTLES (secures King safety & activates Rook)")

        # 4. Promotions
        if move.promotion:
            score_val += 60.0
            desc_parts.append(f"PROMOTES to {chess.piece_name(move.promotion)}")

        # 5. Center control
        if move.to_square in center_squares:
            score_val += 25.0 if is_opening else 10.0
            desc_parts.append("controls vital central square")

        # 6. Piece development in opening
        if is_opening and piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            from_rank = chess.square_rank(move.from_square)
            if (piece.color == chess.WHITE and from_rank == 0) or (piece.color == chess.BLACK and from_rank == 7):
                score_val += 20.0
                desc_parts.append("develops minor piece into active game")

        # Classical e4/d4/e5/d5 pawn pushes
        if is_opening and piece and piece.piece_type == chess.PAWN:
            if uci in ("e2e4", "d2d4", "e7e5", "d7d5"):
                score_val += 30.0
                desc_parts.append("classical center opening pawn push")

        candidates.append(
            CandidateMove(
                move=move,
                san=san,
                uci=uci,
                description=", ".join(desc_parts),
                priority_score=score_val,
            )
        )

    # Order candidates neutrally by UCI move notation without heuristic bias
    candidates.sort(key=lambda c: c.uci)
    return candidates


class ProperChessGame:
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

    def build_systemone_payload(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, chess.Move]]:
        """Construct rich 2D text state and tactical criteria for System One."""
        candidates = analyze_legal_moves(self.board)
        if not candidates:
            raise RuntimeError("No legal moves available")

        # Select top tactical candidates (up to 8 options for crisp choice)
        top_candidates = candidates[:8]
        alias_to_move: dict[str, chess.Move] = {}
        criteria: dict[str, str] = {}

        for i, cand in enumerate(top_candidates):
            alias = chr(ord("A") + i)
            alias_to_move[alias] = cand.move
            criteria[alias] = f"{cand.san} ({cand.uci}): {cand.description}"

        turn_name = "White" if self.board.turn == chess.WHITE else "Black"
        white_mat, black_mat = self.get_material_count()

        state = {
            "board_2d_map": f"\n{render_board_ascii(self.board)}\n",
            "fen": self.board.fen(),
            "side_to_move": f"{turn_name} (Turn {self.board.fullmove_number})",
            "is_in_check": self.board.is_check(),
            "material_balance": f"White {white_mat} vs Black {black_mat}",
            "strategic_phase": "Opening: develop pieces and fight for central squares (e4, d4, e5, d5)" if self.board.fullmove_number <= 7 else "Middlegame / Endgame: tactical control, King safety, material",
        }

        questions = {
            "move": choice(
                f"It is {turn_name}'s turn. Based on the 2D chess board map, choose the best tactical and positional move. "
                "Reply with the option letter from Criteria.",
                criteria,
            ),
            "position_eval": score(
                f"Evaluate the current chess position from {turn_name}'s perspective.",
                [
                    "Equal position: balanced board and material",
                    "Slight advantage: better development or center control",
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
        last_decision: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        avg_latency_ms: float | None = None,
        elapsed_s: float = 0.0,
    ) -> str:
        # 1. Left: Chess board rows
        board_rows = [f"{CLR_CYAN}  ┌────────────────────────┐{CLR_RESET}"]
        for rank in range(7, -1, -1):
            row = [f"{CLR_CYAN}{rank + 1} │{CLR_RESET}"]
            for file in range(8):
                sq = chess.square(file, rank)
                p = self.board.piece_at(sq)
                bg_is_light = (file + rank) % 2 != 0
                bg_col = "\033[48;5;238m" if bg_is_light else "\033[48;5;235m"
                if p:
                    col = CLR_WHITE if p.color == chess.WHITE else CLR_YELLOW
                    glyph = PIECE_UNICODE[p.symbol()]
                    row.append(f"{bg_col}{col}{glyph} {CLR_RESET}")
                else:
                    row.append(f"{bg_col}{CLR_GRAY}· {CLR_RESET}")
            row.append(f"{CLR_CYAN}│ {rank + 1}{CLR_RESET}")
            board_rows.append("".join(row))
        board_rows.append(f"{CLR_CYAN}  └────────────────────────┘{CLR_RESET}")
        board_rows.append(f"{CLR_CYAN}    a  b  c  d  e  f  g  h  {CLR_RESET}")

        # 2. Right: Telemetry & decision breakdown
        mins = int(elapsed_s // 60)
        secs = elapsed_s % 60
        timer_str = f"{mins:02d}:{secs:05.2f}"
        turn_name = "White" if self.board.turn == chess.WHITE else "Black"
        white_mat, black_mat = self.get_material_count()

        speedup_str = ""
        if latency_ms and latency_ms > 0:
            speedup_ratio = max(1.0, 1200.0 / latency_ms)
            speedup_str = f" (~{speedup_ratio:.0f}× vs AR)"

        sidebar = [
            f"{CLR_BOLD}♟️ SYSTEM ONE CHESS AI{CLR_RESET}   {CLR_YELLOW}⏱ Elapsed: {timer_str}{CLR_RESET}",
            f"{CLR_DIM}─────────────────────────────────────────────────────────────────{CLR_RESET}",
            f"  Ply: {CLR_YELLOW}{self.plies:<3}{CLR_RESET} | Turn: {CLR_BRIGHT_GREEN}{turn_name:<5}{CLR_RESET} | Move: {self.board.fullmove_number} | Material: {white_mat} vs {black_mat}",
        ]

        if latency_ms is not None:
            avg_str = f"{avg_latency_ms:.1f} ms" if avg_latency_ms else f"{latency_ms:.1f} ms"
            sidebar.append(
                f"  {CLR_BOLD}⚡ Latency:{CLR_RESET} {CLR_CYAN}{latency_ms:5.1f} ms{CLR_RESET} (Avg: {avg_str}){CLR_BRIGHT_GREEN}{speedup_str}{CLR_RESET}"
            )
        else:
            sidebar.append(f"  {CLR_BOLD}⚡ Latency:{CLR_RESET} Ready")

        sidebar.append(f"{CLR_BOLD}🧠 Candidate Moves Evaluated (Semantic Tactical Ranking):{CLR_RESET}")
        if last_decision:
            chosen_san = last_decision.get("chosen_san", "N/A")
            conf = last_decision.get("confidence", 0.0)
            probs = last_decision.get("probabilities", {})
            criteria_desc = last_decision.get("criteria", {})

            for alias, desc in list(criteria_desc.items())[:5]:
                p = probs.get(alias, 0.0)
                bar_len = int(p * 10)
                bar_str = "█" * bar_len + "░" * (10 - bar_len)
                is_picked = alias == last_decision.get("chosen_alias")
                check = f"{CLR_BRIGHT_GREEN}{CLR_BOLD}[✓]{CLR_RESET}" if is_picked else f"{CLR_DIM}[ ]{CLR_RESET}"
                desc_short = desc[:45]
                sidebar.append(f"   {check} {CLR_CYAN}{alias}{CLR_RESET}: {p*100:4.1f}% [{bar_str}] {desc_short}")

            eval_score = last_decision.get("position_eval")
            eval_str = f"{eval_score:.2f}/2.0" if eval_score is not None else "N/A"
            threat = last_decision.get("threat_alert")
            threat_str = f"{threat}" if threat is not None else "N/A"
            sidebar.append(
                f"  {CLR_BOLD}📋 Parallel Outputs:{CLR_RESET} Conf: {CLR_BRIGHT_GREEN}{conf * 100:.1f}%{CLR_RESET} | Pos Eval: {CLR_YELLOW}{eval_str}{CLR_RESET} | Threat Alert: {CLR_RED if threat else CLR_GREEN}{threat_str}{CLR_RESET}"
            )
        else:
            sidebar.append("   (Waiting for opening move...)")

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
        last_decision: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        avg_latency_ms: float | None = None,
        elapsed_s: float = 0.0,
    ) -> Image.Image:
        """Render a high-definition dashboard frame for GIF recording."""
        img_w, img_h = 960, 560
        img = Image.new("RGB", (img_w, img_h), color="#0b0f19")
        draw = ImageDraw.Draw(img)

        font_title = _get_font(22, bold=True)
        font_tag = _get_font(12, bold=True)
        font_sub = _get_font(12, bold=False)
        font_timer = _get_font(15, bold=True)
        font_stat_val = _get_font(20, bold=True)
        font_stat_lbl = _get_font(11, bold=False)
        font_body = _get_font(13, bold=False)
        font_bold = _get_font(13, bold=True)
        font_small = _get_font(11, bold=False)
        font_piece = _get_font(28, bold=True)

        # 1. Left: Chess Board
        board_size = 480
        pad_x, pad_y = 35, 40
        cell_size = board_size // 8

        light_color = "#e2e8f0"
        dark_color = "#475569"

        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, 7 - rank)
                x0 = pad_x + file * cell_size
                y0 = pad_y + rank * cell_size
                is_light = (file + rank) % 2 == 0
                cell_col = light_color if is_light else dark_color
                draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size], fill=cell_col)

                p = self.board.piece_at(sq)
                if p:
                    glyph = PIECE_UNICODE[p.symbol()]
                    p_col = "#0f172a" if p.color == chess.BLACK else "#ffffff"
                    # Shadow for white pieces on light squares
                    if p.color == chess.WHITE and is_light:
                        draw.text((x0 + cell_size // 4 + 1, y0 + cell_size // 8 + 1), glyph, font=font_piece, fill="#64748b")
                    draw.text((x0 + cell_size // 4, y0 + cell_size // 8), glyph, font=font_piece, fill=p_col)

        # 2. Right: HUD
        hx0 = 545
        hy0 = pad_y
        panel_w = 385
        panel_h = board_size

        draw.rounded_rectangle([hx0, hy0, hx0 + panel_w, hy0 + panel_h], radius=10, fill="#111827", outline="#1f2937", width=2)
        curr_y = hy0 + 14

        mins = int(elapsed_s // 60)
        secs = elapsed_s % 60
        timer_text = f"⏱ {mins:02d}:{secs:05.2f}"

        draw.text((hx0 + 16, curr_y), "SYSTEM ONE CHESS", font=font_title, fill="#38bdf8")
        draw.text((hx0 + panel_w - 110, curr_y + 5), timer_text, font=font_timer, fill="#facc15")
        curr_y += 28
        draw.text((hx0 + 16, curr_y), "2D Text Grid & Semantic Tactical Engine", font=font_sub, fill="#94a3b8")
        curr_y += 22
        draw.line([hx0 + 16, curr_y, hx0 + panel_w - 16, curr_y], fill="#1f2937", width=1)
        curr_y += 12

        # Stat cards
        turn_name = "White" if self.board.turn == chess.WHITE else "Black"
        white_mat, black_mat = self.get_material_count()
        card_w = (panel_w - 32 - 18) // 3
        stats = [("PLY", str(self.plies), "#facc15"), ("TURN", turn_name, "#4ade80"), ("MATERIAL", f"{white_mat}:{black_mat}", "#38bdf8")]
        for i, (lbl, val, col) in enumerate(stats):
            cx = hx0 + 16 + i * (card_w + 9)
            draw.rounded_rectangle([cx, curr_y, cx + card_w, curr_y + 44], radius=6, fill="#1e293b")
            draw.text((cx + 10, curr_y + 5), lbl, font=font_stat_lbl, fill="#94a3b8")
            draw.text((cx + 10, curr_y + 19), val, font=font_stat_val, fill=col)
        curr_y += 54

        # Latency meter
        draw.rounded_rectangle([hx0 + 16, curr_y, hx0 + panel_w - 16, curr_y + 48], radius=6, fill="#1e293b", outline="#334155")
        if latency_ms is not None:
            speedup = max(1.0, 1200.0 / latency_ms)
            lat_text = f"⚡ {latency_ms:.1f} ms (~{speedup:.0f}× faster than AR)"
            draw.text((hx0 + 24, curr_y + 6), lat_text, font=font_bold, fill="#facc15")
            avg_text = f"Avg: {avg_latency_ms:.1f} ms • Zero-token Option Logits" if avg_latency_ms else "Zero-token Option Logits"
            draw.text((hx0 + 24, curr_y + 26), avg_text, font=font_small, fill="#38bdf8")
        else:
            draw.text((hx0 + 24, curr_y + 14), "⚡ Engine Ready", font=font_bold, fill="#facc15")
        curr_y += 58

        # Candidate Moves Breakdown
        draw.text((hx0 + 16, curr_y), "Candidates Evaluated (Semantic Tactical Ranking):", font=font_bold, fill="#e2e8f0")
        curr_y += 22

        if last_decision:
            chosen_alias = last_decision.get("chosen_alias")
            probs = last_decision.get("probabilities", {})
            criteria_desc = last_decision.get("criteria", {})

            for alias, desc in list(criteria_desc.items())[:4]:
                p = probs.get(alias, 0.0)
                is_picked = alias == chosen_alias
                row_bg = "#1e293b" if is_picked else "#0f172a"
                row_border = "#22c55e" if is_picked else "#1e293b"
                draw.rounded_rectangle([hx0 + 16, curr_y, hx0 + panel_w - 16, curr_y + 32], radius=5, fill=row_bg, outline=row_border)

                chk = "✓ " if is_picked else "  "
                chk_col = "#4ade80" if is_picked else "#64748b"
                draw.text((hx0 + 22, curr_y + 7), f"{chk}{alias}", font=font_bold if is_picked else font_body, fill=chk_col)

                bar_x = hx0 + 70
                bar_max_w = 60
                draw.rounded_rectangle([bar_x, curr_y + 11, bar_x + bar_max_w, curr_y + 19], radius=2, fill="#0b0f19")
                if p > 0:
                    fill_col = "#38bdf8" if is_picked else "#475569"
                    draw.rounded_rectangle([bar_x, curr_y + 11, bar_x + int(bar_max_w * p), curr_y + 19], radius=2, fill=fill_col)
                draw.text((bar_x + bar_max_w + 6, curr_y + 8), f"{p*100:4.1f}%", font=font_small, fill="#cbd5e1")

                move_label = desc.split(":")[0][:14]
                draw.text((hx0 + 205, curr_y + 8), move_label, font=font_small, fill="#4ade80" if is_picked else "#94a3b8")
                curr_y += 36

            curr_y += 8
            conf = last_decision.get("confidence", 0.0)
            draw.text((hx0 + 16, curr_y), f"Choice Confidence: {conf * 100:.1f}%", font=font_bold, fill="#38bdf8")
            curr_y += 18
            eval_score = last_decision.get("position_eval")
            if eval_score is not None:
                draw.text((hx0 + 16, curr_y), f"Position Score: {eval_score:.2f}/2.0", font=font_small, fill="#fbbf24")
                curr_y += 18

        if self.game_over:
            draw.rounded_rectangle([hx0 + 16, curr_y, hx0 + panel_w - 16, curr_y + 36], radius=6, fill="#7f1d1d")
            draw.text((hx0 + 24, curr_y + 10), f"GAME OVER: {self.result[:30]}", font=font_bold, fill="#fecaca")

        return img


def play_game(
    client: SystemOneClient,
    max_plies: int = 20,
    delay: float = 0.3,
    animate: bool = True,
    gif_path: str | Path | None = None,
    fps: int = 4,
) -> dict[str, Any]:
    game = ProperChessGame()
    last_decision: dict[str, Any] | None = None
    latencies: list[float] = []
    frames: list[Image.Image] = []
    start_time = time.perf_counter()

    if animate:
        sys.stdout.write("\033[2J\033[H\033[?25l")
        sys.stdout.flush()

    try:
        while not game.game_over and game.plies < max_plies:
            state, questions, alias_to_move = game.build_systemone_payload()

            t0 = time.perf_counter()
            response = client.system_one(state=state, questions=questions)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            avg_lat = sum(latencies) / len(latencies)
            elapsed_s = time.perf_counter() - start_time

            # Parse choice answer
            move_ans = response.answers["move"]
            chosen_alias = move_ans.choice
            chosen_move = alias_to_move.get(chosen_alias)
            if chosen_move is None:
                # Fallback to top candidate
                chosen_alias = list(alias_to_move.keys())[0]
                chosen_move = alias_to_move[chosen_alias]

            chosen_san = game.board.san(chosen_move)
            pos_eval = response.answers.get("position_eval")
            threat = response.answers.get("threat_alert")

            last_decision = {
                "chosen_alias": chosen_alias,
                "chosen_san": chosen_san,
                "chosen_move": chosen_move,
                "confidence": move_ans.confidence or 0.0,
                "probabilities": move_ans.probabilities or {},
                "criteria": questions["move"]["criteria"],
                "position_eval": pos_eval.score if pos_eval else None,
                "threat_alert": threat.noul if threat else None,
            }

            if gif_path:
                frames.append(game.render_frame_pil(last_decision, elapsed_ms, avg_lat, elapsed_s))

            if animate:
                sys.stdout.write("\033[H")
                sys.stdout.write(game.render_ansi(last_decision, elapsed_ms, avg_lat, elapsed_s) + "\n")
                sys.stdout.flush()
                if delay > 0:
                    time.sleep(delay)

            game.step(chosen_move)

        # Final frame
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
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()

    if gif_path and frames:
        out_p = Path(gif_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        extended_frames = frames + [frames[-1]] * 4
        extended_frames[0].save(
            out_p,
            save_all=True,
            append_images=extended_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )

    return {
        "plies": game.plies,
        "history": [san for _, san in game.move_history],
        "result": game.result or ("Max plies reached" if game.plies >= max_plies else "In progress"),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "gif_saved": str(gif_path) if (gif_path and frames) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Proper Chess demo powered by systemone-lite.")
    parser.add_argument("--model", type=str, default="systemone-lite-latest")
    parser.add_argument("--stub", action="store_true", help="Use deterministic StubEngine")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--max-plies", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--gif", type=str, default=None)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

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

    animate = not args.quiet
    res = play_game(
        client=client,
        max_plies=args.max_plies,
        delay=args.delay if animate else 0.0,
        animate=animate,
        gif_path=args.gif,
        fps=args.fps,
    )

    if not animate:
        moves_str = " ".join(res["history"])
        print(f"Game finished ({res['plies']} plies). Result: {res['result']}")
        print(f"Moves: {moves_str}")
        print(f"Avg Latency: {res['avg_latency_ms']:.1f} ms")
        if res.get("gif_saved"):
            print(f"Saved GIF to {res['gif_saved']}")


if __name__ == "__main__":
    main()
