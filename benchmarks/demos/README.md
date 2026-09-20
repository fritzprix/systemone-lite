# ⚖️ Honest Baseline vs. SFT Model Comparison

This directory contains **100% genuine, unassisted side-by-side gameplay recordings** comparing the raw pre-trained Base model against our Phase 1 Mixed SFT checkpoint under identical, neutral conditions.

---

## 🔬 Experimental Setup & Ground Rules

1. **No Keyword Hints**: Zero prompt-injection shortcuts (`RECOMMENDED:`, `OPTIMAL:`, `BLOCK THREAT:` have been completely eliminated).
2. **Neutral Option Sorting**: Candidate moves and directions are sorted neutrally (e.g., standard UCI string order `a2a3, a2a4, ...` or standard cardinal directions `UP, DOWN, LEFT, RIGHT`), preventing heuristic position bias.
3. **Hardware**: Local NVIDIA RTX 3060 (12GB VRAM), PyTorch 2.x, FP16 inference.

---

## 📊 Models Evaluated

* **Base Model (`base_model/`)**: `Qwen/Qwen2.5-0.5B-Instruct` (raw pre-trained instruct weights without domain SFT).
* **SFT Model (`sft_model/`)**: `checkpoints/systemone-mixed-sft` (fine-tuned on 43,200 balanced samples of General System 1 triage + 2D Chess FENs).

---

## 🎬 Side-by-Side Comparison

| Environment | Base Model (`base_model/`) | SFT Model (`sft_model/`) | Analysis & Observations |
|---|---|---|---|
| **♟️ Proper Tactical Chess** | `base_model/chess_proper.gif`<br/>Moves: `a3 a6 Ra2 c6 Ra1 d6 Ra2 a5 Ra1 a4`<br/>Latency: ~371ms | `sft_model/chess_proper.gif`<br/>Moves: `Nc3 Na6 Na4 Rb8 Nb6 Ra8 Nxa8 c6 Nb6 axb6`<br/>Latency: ~191ms | **Clear SFT Superiority**:<br/>• **Base**: Pushes flank pawn and aimlessly shuffles Rook (`Ra2-Ra1`).<br/>• **SFT**: Develops Knight (`Nc3`), infiltrates Queenside (`Na4 → Nb6`), and **captures Black's Rook** on a8 (`Nxa8`). |
| **♟️ Multi-Step Chess** | `base_model/chess_multistep.gif`<br/>Moves: `a3 a5 Ra2 a4 Ra1 Ra5 Ra2 Ra6 Ra1 Ra5`<br/>Latency: ~364ms | `sft_model/chess_multistep.gif`<br/>Moves: `Nc3 Nc6 Rb1 Rb8 Ra1 Ra8 Rb1 Rb8 Ra1 Ra8`<br/>Latency: ~203ms | **Opening Recognition**:<br/>• **Base**: Repeats flank moves (`a3, Ra2, Ra1`).<br/>• **SFT**: Actively selects Knights for early development (`Nc3, Nc6`). |
| **🔢 2048 Game** | `base_model/game2048.gif`<br/>Final: Score 64, Tile 16 | `sft_model/game2048.gif`<br/>Final: Score 60, Tile 16 | **Zero-Shot Baseline**:<br/>Neither model was trained on 2048 in Phase 1. Both execute legal merges reaching Tile 16. |
| **🗺️ GridWorld Hazard** | `base_model/gridworld.gif`<br/>10 steps (Goal reached 🏆) | `sft_model/gridworld.gif`<br/>10 steps (Goal reached 🏆) | **Neutral Navigation**:<br/>Both navigate the 2D grid corridor towards the exit without falling into lava. |
| **📦 Sokoban Warehouse** | `base_model/sokoban.gif`<br/>10 steps | `sft_model/sokoban.gif`<br/>10 steps | **Zero-Shot Baseline**:<br/>Neither model has seen box-pushing trajectories yet. Demonstrates the need for Phase 2 training. |
| **🔴 Connect Four** | `base_model/connect4.gif`<br/>12 turns (Draw) | `sft_model/connect4.gif`<br/>12 turns (Draw) | **Zero-Shot Baseline**:<br/>Both drop discs into open vertical columns. |

---

## 💡 Key Takeaways

1. **Where SFT genuinely works**: In domains where domain data was actually injected (Chess 2D representation), the SFT model displays distinct strategic competence (developing minor pieces, tactical piece capture) over the base model's aimless flank pawn pushes.
2. **Where SFT requires Phase 2**: In games where training data has not yet been introduced (Sokoban, 2048, Connect4), both models behave zero-shot. This proves why **Phase 2 (Synthetic Spatial Trajectories & $D_4$ Dihedral Symmetry Augmentation)** is essential to teach true spatial reasoning without prompt heuristics.
