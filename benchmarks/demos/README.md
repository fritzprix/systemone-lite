# Bare-face demos (Base / Phase 1 / Phase 2)

Side-by-side recordings under **stripped prompts** (no tactical keyword hints, no solver overrides).

Regenerate:

```bash
python scripts/run_bare_demos.py                 # all tags
python scripts/run_bare_demos.py --only spatial_v2_s1
# → benchmarks/demos/{base_model,sft_model,spatial_v2,spatial_v2b,spatial_v2_s1}/*.gif
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
| Models | Base = `Qwen/Qwen2.5-0.5B-Instruct`. Phase 1 = `systemone-mixed-sft`. Phase 2 gate = `systemone-spatial-v2`. Continual attempt = `systemone-spatial-v2b` (20k from base, **not** resume from v2). S1 cloze continue = `systemone-spatial-v2-s1` (20k from v2 + cloze/diversity mix). |

Machine-readable numbers: [`bare_face_report.json`](bare_face_report.json).  
Held-out gates: [`../spatial_v2_report.json`](../spatial_v2_report.json) (v2) · [`../spatial_v2b_report.json`](../spatial_v2b_report.json) (v2b).

## Where to watch

| Folder | Model |
|---|---|
| [`base_model/`](base_model/) | Base 0.5B |
| [`sft_model/`](sft_model/) | Phase 1 mixed SFT |
| [`spatial_v2/`](spatial_v2/) | Phase 2 spatial SFT (51.2k, gate PASS on prior eval) |
| [`spatial_v2b/`](spatial_v2b/) | 20k cold-start attempt (gate FAIL on current eval) |
| [`spatial_v2_s1/`](spatial_v2_s1/) | v2 + cloze/diversity continue (20k); JevBench Acc 49.8% |

Each folder has: `chess_proper.gif`, `chess_multistep.gif`, `game2048.gif`, `gridworld.gif`, `sokoban.gif`, `connect4.gif`.

## Snapshot

| Env | Base | Phase 1 | `spatial_v2` | `spatial_v2b` |
|---|---|---|---|---|
| Chess proper | `a3 …` rook loop | `Nc3 …` then rook loop | `d4 …` more center | `b4 Nc6 …` (short clip) |
| 2048 | score 32 / max 8 | score 36 / max 8 | score **60** / max **16** | score 60 / max 16 |
| GridWorld | not solved | not solved | not solved | died early |
| Sokoban | not solved | not solved | not solved | not solved |
| Connect4 | no winner | no winner | no winner | no winner |

GIFs are illustrations; **JSON held-out metrics** are the claim surface.
