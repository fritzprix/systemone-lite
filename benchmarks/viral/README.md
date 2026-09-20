# 🎬 systemone-lite Viral Demo Gallery

This directory contains animated visual demos comparing different iterations of `systemone-lite`.

---

## 🚀 Version 1.0: `systemone-mixed-sft` (Latest Phase 1 Model)
> **Model Checkpoint**: `checkpoints/systemone-mixed-sft`  
> **Backbone**: `Qwen2.5-0.5B-Instruct`  
> **Training**: 43,200 samples balanced co-training (Chess 50:50 unbiased + General SFT)  
> **Latency**: ~30ms – 40ms per decision (~17× to 35× faster than Autoregressive CoT)

| Game / Demo | Preview | Specifications | Key Highlights |
|---|:---:|---|---|
| **♟️ Multi-Step Chess** | `viral/v1_mixed_sft/chess_multistep.gif` | 23 frames / 11 plies<br/>~70ms / turn | 2-Stage pipeline: Step 1 piece selection → Step 2 move selection |
| **♟️ Proper Chess (Tactical)** | `viral/v1_mixed_sft/chess_proper.gif` | 20 plies<br/>~65ms / ply | Semantic move ranking, positional evaluation & threat alerting |
| **🔢 2048 Agent** | `viral/v1_mixed_sft/game2048.gif` | 25 steps<br/>Score: 148, Max: 32<br/>~35ms / move | 4-4, 8-8, 16-16 strategic directional merges |
| **🗺️ GridWorld Hazard** | `viral/v1_mixed_sft/gridworld.gif` | 10 steps (Goal reached 🏆)<br/>~36ms / step | Real-time lava (🔥) hazard avoidance & shortest path navigation |
| **📦 Sokoban AI** | `viral/v1_mixed_sft/sokoban.gif` | 15 steps<br/>~33ms / step | Raw 2D newline map parsing, legal action masking, box delivered to goal (✅) |
| **🔴 Connect Four** | `viral/v1_mixed_sft/connect4.gif` | 18 turns<br/>~39ms / turn | 7-column parallel probability HUD and strategic vertical stacking |

---

## 🏛️ Version 0.x: Legacy Baselines
> Early checkpoints prior to unified mixed co-training.

* **`v0_legacy/snake_demo.gif`**: Autonomous Snake navigation trained on initial general-only SFT (`dwidlee/systemone-lite-0.5b`).
* **`v0_legacy/systemone_lite_chess.gif`**: Initial prototype chess policy.
