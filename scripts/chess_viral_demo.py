#!/usr/bin/env python3
"""Generate a short viral-style chess demo video powered by systemone-lite.

Flow per turn (when many legal moves):
  1) choice: which piece to move?  (legal origins only)
  2) choice: where to move it?     (legal targets for that piece)

Renders cinematic frames → MP4 (and optional GIF).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from systemone_lite import SystemOneClient, choice
from systemone_lite.chess_data import board_state
from systemone_lite.infer import reset_engine, set_default_model
from systemone_lite.stub import StubEngine

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "benchmarks" / "viral"

# Social-friendly portrait-ish canvas (also crops fine to 1:1 / 9:16 later)
W, H = 1080, 1350

PIECE_GLYPH = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansSymbols2-Regular.otf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _piece_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansSymbols2-Regular.otf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return _font(size)


FONT_TITLE = _font(54)
FONT_SUB = _font(28)
FONT_BODY = _font(26)
FONT_SMALL = _font(22)
FONT_PIECE = _piece_font(64)
FONT_HUGE = _font(72)


def board_to_state(board: chess.Board) -> dict:
    return board_state(board)


def legal_by_origin(board: chess.Board) -> dict[str, list[chess.Move]]:
    grouped: dict[str, list[chess.Move]] = {}
    for move in board.legal_moves:
        origin = chess.square_name(move.from_square)
        grouped.setdefault(origin, []).append(move)
    return grouped


def describe_piece(board: chess.Board, square_name: str) -> str:
    square = chess.parse_square(square_name)
    piece = board.piece_at(square)
    if piece is None:
        return square_name
    color = "white" if piece.color == chess.WHITE else "black"
    desc = f"{color} {chess.piece_name(piece.piece_type)} on {square_name}"
    file = chess.square_file(square)
    if file in (3, 4):
        desc += " (central piece)"
    return desc


def _score_origin(board: chess.Board, sq_name: str) -> float:
    sq = chess.parse_square(sq_name)
    piece = board.piece_at(sq)
    if piece is None:
        return 0.0
    s = 0.0
    file = chess.square_file(sq)
    if file in (3, 4):  # d or e file
        s += 25.0
    elif file in (2, 5):  # c or f file
        s += 12.0
    if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
        s += 18.0
    return s


def _score_target(board: chess.Board, move: chess.Move) -> float:
    s = 0.0
    if board.gives_check(move):
        s += 50.0
    if board.is_capture(move):
        s += 35.0
    if move.to_square in (chess.E4, chess.D4, chess.E5, chess.D5):
        s += 25.0
    return s


def _alias_criteria(options: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Map arbitrary option keys to single-token aliases (A, B, C, ...).

    Qwen tokenizes chess squares / UCI as multi-token; first-token scoring then
    collapses distinct moves. Aliases keep option-constrained softmax valid.
    """
    alias_to_key: dict[str, str] = {}
    criteria: dict[str, str] = {}
    for i, (key, desc) in enumerate(options.items()):
        if i < 26:
            alias = chr(ord("A") + i)
        else:
            alias = str(i)
        alias_to_key[alias] = key
        criteria[alias] = f"{key}: {desc}"
    return criteria, alias_to_key


def decide_move(client: SystemOneClient, board: chess.Board) -> tuple[chess.Move, dict]:
    """Two-step System One decision → legal chess.Move + debug payload."""
    grouped = legal_by_origin(board)
    if not grouped:
        raise RuntimeError("no legal moves")

    state = board_to_state(board)

    # Sort origins by tactical/central priority instead of alphabetical
    origin_keys = sorted(grouped.keys(), key=lambda sq: _score_origin(board, sq), reverse=True)
    origin_raw = {sq: describe_piece(board, sq) for sq in origin_keys}
    origin_criteria, origin_alias = _alias_criteria(origin_raw)
    step1 = client.system_one(
        state=state,
        questions={
            "piece": choice(
                "Pick one piece that should move this turn. Prefer development, "
                "center control, and king safety. Reply with the option letter.",
                origin_criteria,
            )
        },
    )
    origin_alias_chosen = step1.answers["piece"].choice
    origin = origin_alias[origin_alias_chosen]
    origin_probs = {
        origin_alias[k]: v for k, v in step1.answers["piece"].probabilities.items()
    }
    origin_conf = step1.answers["piece"].confidence

    # Sort target moves by tactical priority (checks, captures, center control)
    targets = sorted(grouped[origin], key=lambda m: _score_target(board, m), reverse=True)
    target_raw: dict[str, str] = {}
    uci_to_move: dict[str, chess.Move] = {}
    for move in targets:
        key = move.uci()
        to_sq = chess.square_name(move.to_square)
        captured = board.piece_at(move.to_square)
        note = f"to {to_sq}"
        if captured:
            note += f", capture {chess.piece_name(captured.piece_type)}"
        if move.promotion:
            note += f", promote to {chess.piece_name(move.promotion)}"
        if move.to_square in (chess.E4, chess.D4, chess.E5, chess.D5):
            note += ", controls center"
        target_raw[key] = note
        uci_to_move[key] = move

    move_criteria, move_alias = _alias_criteria(target_raw)
    step2 = client.system_one(
        state={
            **state,
            "selected_piece": origin,
            "selected_desc": origin_raw[origin],
        },
        questions={
            "move": choice(
                f"The piece on {origin} is selected. Choose the best legal "
                "destination. Reply with the option letter.",
                move_criteria,
            )
        },
    )
    move_alias_chosen = step2.answers["move"].choice
    uci = move_alias[move_alias_chosen]
    move = uci_to_move[uci]
    move_probs = {
        move_alias[k]: v for k, v in step2.answers["move"].probabilities.items()
    }
    return move, {
        "origin": origin,
        "origin_probs": origin_probs,
        "origin_confidence": origin_conf,
        "move_uci": uci,
        "move_probs": move_probs,
        "move_confidence": step2.answers["move"].confidence,
        "latency_hint_ms": None,
    }


def _draw_rounded_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    radius: int = 24,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill)


def render_board(
    draw: ImageDraw.ImageDraw,
    board: chess.Board,
    *,
    left: int,
    top: int,
    size: int,
    last_move: chess.Move | None = None,
    highlight_from: str | None = None,
    highlight_to: str | None = None,
) -> None:
    light = (238, 238, 210)
    dark = (118, 150, 86)
    hi_from = (246, 246, 105)
    hi_to = (186, 202, 68)
    cell = size // 8

    for rank in range(8):
        for file in range(8):
            sq = chess.square(file, 7 - rank)
            name = chess.square_name(sq)
            x0 = left + file * cell
            y0 = top + rank * cell
            base = light if (file + rank) % 2 == 0 else dark
            if last_move and sq in (last_move.from_square, last_move.to_square):
                base = hi_from if sq == last_move.from_square else hi_to
            if highlight_from and name == highlight_from:
                base = hi_from
            if highlight_to and name == highlight_to:
                base = hi_to
            draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=base)

            piece = board.piece_at(sq)
            if piece:
                glyph = PIECE_GLYPH[piece.symbol()]
                bbox = draw.textbbox((0, 0), glyph, font=FONT_PIECE)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                color = (20, 20, 20) if piece.color == chess.WHITE else (15, 15, 15)
                # slight outline for visibility on both squares
                cx = x0 + (cell - tw) // 2
                cy = y0 + (cell - th) // 2 - 4
                draw.text((cx, cy), glyph, font=FONT_PIECE, fill=color)

    # coordinates
    for i, file in enumerate("abcdefgh"):
        draw.text(
            (left + i * cell + cell // 2 - 6, top + size + 8),
            file,
            font=FONT_SMALL,
            fill=(180, 180, 180),
        )
    for i, rank in enumerate("87654321"):
        draw.text(
            (left - 28, top + i * cell + cell // 2 - 10),
            rank,
            font=FONT_SMALL,
            fill=(180, 180, 180),
        )


def draw_prob_bars(
    draw: ImageDraw.ImageDraw,
    *,
    title: str,
    probs: dict[str, float],
    chosen: str,
    x: int,
    y: int,
    width: int,
    max_items: int = 5,
) -> int:
    draw.text((x, y), title, font=FONT_BODY, fill=(230, 230, 230))
    y += 40
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:max_items]
    for key, p in items:
        bar_w = int(width * p)
        label = f"{key}  {p:.0%}"
        color = (80, 200, 120) if key == chosen else (70, 90, 120)
        _draw_rounded_rect(draw, (x, y, x + width, y + 36), (40, 44, 56), radius=10)
        if bar_w > 8:
            _draw_rounded_rect(draw, (x, y, x + bar_w, y + 36), color, radius=10)
        draw.text((x + 12, y + 6), label, font=FONT_SMALL, fill=(245, 245, 245))
        y += 48
    return y


def make_base_frame() -> Image.Image:
    img = Image.new("RGB", (W, H), (12, 14, 20))
    draw = ImageDraw.Draw(img)
    # subtle vignette bars
    draw.rectangle([0, 0, W, 110], fill=(18, 22, 32))
    draw.rectangle([0, H - 90, W, H], fill=(18, 22, 32))
    draw.text((48, 28), "systemone-lite", font=FONT_TITLE, fill=(240, 240, 240))
    draw.text(
        (48, 88),
        "local System One  ·  chess decisions  ·  not Jev",
        font=FONT_SMALL,
        fill=(140, 150, 170),
    )
    return img


def frame_title(subtitle: str) -> Image.Image:
    img = make_base_frame()
    draw = ImageDraw.Draw(img)
    draw.text((80, 520), "Typed decisions.", font=FONT_HUGE, fill=(255, 255, 255))
    draw.text((80, 620), "Legal moves only.", font=FONT_HUGE, fill=(120, 220, 160))
    draw.text((80, 760), subtitle, font=FONT_SUB, fill=(180, 190, 210))
    draw.text(
        (80, 1180),
        "github.com/fritzprix/systemone-lite",
        font=FONT_BODY,
        fill=(160, 170, 190),
    )
    return img


def frame_end() -> Image.Image:
    img = make_base_frame()
    draw = ImageDraw.Draw(img)
    draw.text((80, 480), "Run it locally.", font=FONT_HUGE, fill=(255, 255, 255))
    draw.text((80, 580), "MIT · toy project", font=FONT_SUB, fill=(140, 220, 170))
    draw.text(
        (80, 700),
        "pip install -e .\nsystemone-lite --port 8000",
        font=FONT_BODY,
        fill=(200, 200, 210),
    )
    draw.text(
        (80, 1180),
        "github.com/fritzprix/systemone-lite",
        font=FONT_BODY,
        fill=(160, 170, 190),
    )
    return img


def frame_turn(
    board: chess.Board,
    *,
    ply: int,
    move: chess.Move | None,
    debug: dict | None,
    phase: str,
) -> Image.Image:
    img = make_base_frame()
    draw = ImageDraw.Draw(img)

    turn = "White" if board.turn == chess.WHITE else "Black"
    if move is not None and phase == "after":
        # after push, turn already flipped; show mover as opposite
        turn = "Black" if board.turn == chess.WHITE else "White"

    draw.text(
        (48, 130),
        f"Ply {ply}  ·  {phase}  ·  {turn}",
        font=FONT_SUB,
        fill=(200, 210, 230),
    )

    board_size = 820
    board_left = (W - board_size) // 2
    board_top = 190
    render_board(
        draw,
        board,
        left=board_left,
        top=board_top,
        size=board_size,
        last_move=move if phase == "after" else None,
        highlight_from=debug.get("origin") if debug and phase == "decide" else None,
        highlight_to=(
            chess.square_name(move.to_square)
            if move is not None and phase == "decide"
            else None
        ),
    )

    panel_top = board_top + board_size + 50
    _draw_rounded_rect(
        draw,
        (40, panel_top, W - 40, H - 110),
        (24, 28, 40),
        radius=28,
    )

    if debug and phase in {"decide", "after"}:
        y = panel_top + 28
        draw.text(
            (70, y),
            f"choice → {debug['move_uci']}   conf {debug['move_confidence']:.2f}",
            font=FONT_BODY,
            fill=(120, 220, 160),
        )
        y += 50
        y = draw_prob_bars(
            draw,
            title="piece probs",
            probs=debug["origin_probs"],
            chosen=debug["origin"],
            x=70,
            y=y,
            width=440,
            max_items=4,
        )
        draw_prob_bars(
            draw,
            title="move probs",
            probs=debug["move_probs"],
            chosen=debug["move_uci"],
            x=560,
            y=panel_top + 78,
            width=440,
            max_items=4,
        )
    else:
        draw.text(
            (70, panel_top + 40),
            "Asking System One…",
            font=FONT_BODY,
            fill=(180, 190, 210),
        )

    return img


def hold(frames: list[Image.Image], img: Image.Image, seconds: float, fps: int) -> None:
    n = max(1, int(round(seconds * fps)))
    frames.extend([img] * n)


def run_demo(
    *,
    plies: int,
    fps: int,
    use_stub: bool,
    model: str,
    out_mp4: Path,
    out_gif: Path | None,
    hold_title: float,
    hold_think: float,
    hold_decide: float,
    hold_after: float,
    hold_end: float,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reset_engine()
    if use_stub:
        client = SystemOneClient(engine=StubEngine())
        model_label = "stub"
    else:
        set_default_model(model)
        client = SystemOneClient(model=model)
        model_label = Path(model).name if "/" in model or model.startswith(".") else model

    board = chess.Board()
    frames: list[Image.Image] = []

    hold(
        frames,
        frame_title(f"{model_label}  ·  legal criteria  ·  both sides System One"),
        hold_title,
        fps,
    )

    for ply in range(1, plies + 1):
        if board.is_game_over():
            break

        thinking = frame_turn(board, ply=ply, move=None, debug=None, phase="thinking")
        hold(frames, thinking, hold_think, fps)

        t0 = time.perf_counter()
        move, debug = decide_move(client, board)
        debug["latency_hint_ms"] = round((time.perf_counter() - t0) * 1000)

        decide = frame_turn(board, ply=ply, move=move, debug=debug, phase="decide")
        hold(frames, decide, hold_decide, fps)

        board.push(move)
        after = frame_turn(board, ply=ply, move=move, debug=debug, phase="after")
        hold(frames, after, hold_after, fps)

    # Result banner if game ended early.
    if board.is_game_over():
        end = frame_end()
        draw = ImageDraw.Draw(end)
        draw.text(
            (80, 420),
            f"Result: {board.result()}  ({board.fullmove_number} moves)",
            font=FONT_SUB,
            fill=(255, 200, 120),
        )
        hold(frames, end, hold_end, fps)
    else:
        hold(frames, frame_end(), hold_end, fps)

    # Write MP4
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        out_mp4,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))
    finally:
        writer.close()

    duration_s = len(frames) / max(fps, 1)
    print(f"wrote {out_mp4} ({len(frames)} frames @ {fps}fps ≈ {duration_s:.1f}s)")

    if out_gif is not None:
        gif_frames = []
        # Keep GIF under ~200 frames for shareability.
        step = max(1, len(frames) // 180)
        for i, frame in enumerate(frames):
            if i % step != 0:
                continue
            small = frame.resize((540, 675), Image.Resampling.LANCZOS)
            gif_frames.append(np.asarray(small.convert("RGB")))
        imageio.mimsave(out_gif, gif_frames, fps=max(4, fps // max(step, 1)), loop=0)
        print(f"wrote {out_gif} ({len(gif_frames)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Viral chess demo for systemone-lite")
    parser.add_argument("--plies", type=int, default=20, help="Half-moves to play")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--model",
        default="checkpoints/chess-sft",
        help="HF id or local checkpoint (default: fine-tuned chess-sft)",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use StubEngine (no GPU / no weights) for a quick dry-run",
    )
    parser.add_argument(
        "--mp4",
        type=Path,
        default=OUT_DIR / "systemone_lite_chess.mp4",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=OUT_DIR / "systemone_lite_chess.gif",
    )
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--hold-title", type=float, default=3.0)
    parser.add_argument("--hold-think", type=float, default=0.85)
    parser.add_argument("--hold-decide", type=float, default=2.1)
    parser.add_argument("--hold-after", type=float, default=1.35)
    parser.add_argument("--hold-end", type=float, default=3.5)
    args = parser.parse_args()

    run_demo(
        plies=args.plies,
        fps=args.fps,
        use_stub=args.stub,
        model=args.model,
        out_mp4=args.mp4,
        out_gif=None if args.no_gif else args.gif,
        hold_title=args.hold_title,
        hold_think=args.hold_think,
        hold_decide=args.hold_decide,
        hold_after=args.hold_after,
        hold_end=args.hold_end,
    )


if __name__ == "__main__":
    main()
