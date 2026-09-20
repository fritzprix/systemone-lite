# Bare-face Base vs SFT demos

Side-by-side recordings under **stripped prompts** (no tactical keyword hints, no solver overrides).

Regenerate:

```bash
python scripts/run_bare_demos.py
# → benchmarks/demos/{base_model,sft_model}/*.gif
# → benchmarks/demos/bare_face_report.json
```

## Protocol

| Rule | Detail |
|---|---|
| Option text | Bare labels only (`pawn a2->a3`, `Slide UP`, `Move LEFT`). No `CAPTURES` / `CHECK` / `develop` / `RECOMMENDED` / `BLOCKED` / `DEADLY`. |
| Chess option set | **All** legal moves (cap 26 via FEN-seeded subsample). Aliases sorted by UCI. Previous demos used UCI-sort then `[:8]` (a-file bias). |
| Solver override | Disabled. Illegal model output → first legal action only (not BFS / `best_move_*`). |
| Colors | Self-play: both sides are the **same** model. |
| Hardware | Local NVIDIA RTX 3060, FP16 when available. |
| Models | Base = `Qwen/Qwen2.5-0.5B-Instruct`. SFT = `checkpoints/systemone-mixed-sft`. |

Machine-readable numbers: [`bare_face_report.json`](bare_face_report.json).

## Results (this run)

| Env | Base | SFT | Honest read |
|---|---|---|---|
| Chess proper (`chess_proper.gif`) | `a3 a6 Ra2 Ra7 Ra1 Ra8 Ra2 Ra7 Ra1 Ra8` (~380ms) | `Nc3 Na6 Rb1 Rb8 Ra1 Ra8 Rb1 Rb8 Ra1 Ra8` (~157ms) | SFT opens with a knight; then both sides mostly rook-shuffle. **No tactical capture** once `CAPTURES …` tags are gone. |
| Chess multistep | `a3 a5 Ra2 a4 Ra1 Ra5 …` (~70ms) | `Nc3 Nc6 Rb1 Rb8 Ra1 Ra8 …` (~63ms) | Same pattern: early knight recognition on SFT, then rook loops. |
| 2048 | score 32, max tile 8 | score 36, max tile 8 | Near-shot / weak. Phase 1 did not train 2048. |
| GridWorld | 10 steps, **not** solved | 10 steps, **not** solved | Earlier “both reach goal 🏆” was inflated by a **BFS path override** in the demo loop (now removed). |
| Sokoban | 10 steps, not solved | 10 steps, not solved | Zero-shot; no box-pushing trajectories in Phase 1. |
| Connect4 | 12 turns, no winner | 12 turns, no winner | Legal drops only; no win/block policy learned. |

## Takeaways

1. **Chess**: With bare labels + full legal move lists, SFT still prefers opening knights (`Nc3`/`Nc6`) vs Base’s flank pawn / rook shuffle — a real but **narrow** signal. The prior “`Nxa8` rook capture” demo does **not** reproduce without capture keywords.
2. **Other 2D games**: Both models look like zero-shot reflexes. Do not read GIFs as competence proofs.
3. **Held-out numbers** (not GIFs) remain the serious metric: chess move accuracy ~0.05 → ~0.24 on shuffled `board_2d_map` eval (`benchmarks/mixed_vs_base_report.json`).

GIF frames are animated step-by-step (verify with `Image.seek`, not naive `ImageSequence` iteration without `.copy()`).
