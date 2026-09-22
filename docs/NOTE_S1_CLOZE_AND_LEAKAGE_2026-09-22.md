# Note — Zero-leakage eval, NLP cloze, v2→s1 continue (2026-09-22)

## Summary

1. **Train∩eval contamination fixed to 0.00%** (was up to ~91% on debate / ~87% word_games).
2. **`nlp_cloze`** added (WikiText-2, document-disjoint train/test) as a language anchor.
3. Continual SFT **`systemone-spatial-v2-s1`**: 20k steps from `spatial-v2` on the refreshed mix.
4. **JevBench** Acc **49.8%** (was v2 42.9%, Phase‑1 mixed 45.9%). ECE still worse than Phase‑1.

## Dataset hygiene

| Fix | Mechanism |
|---|---|
| `word_games` | Disjoint `TRAIN_*` / `EVAL_*` vocab + plurals |
| `debate_judge` | Held-out topics (train 5 / eval 4 including new motions) |
| Spatial / chess / CA | State-hash rejection vs train when building eval |
| Audit | `scripts/audit_train_eval_overlap.py` (fail if overlap > 0) |
| Rebuild | `scripts/rebuild_heldout_eval.py` |

Helpers: `src/systemone_lite/synth/leakage.py`.

Current local mix (not yet re-uploaded to Hub at note time):

- train **240 800** / eval **4 700**
- includes CA + word_games + **nlp_cloze** (12k / 400)
- text-ish share ≈ **23%** (ticket/alloc/debate/word/cloze) — full “≥50% text S1 pivot” still pending

## NLP cloze

- Module: `src/systemone_lite/synth/nlp_cloze.py`
- Corpus: `Salesforce/wikitext` `wikitext-2-raw-v1` (train→train, test→eval)
- Builder: `scripts/build_synth_diversity.py --cloze-train/--cloze-eval --merge-into-phase2`

## Training

| Run | Init | Steps | Data | Wandb |
|---|---|---:|---|---|
| `spatial-v2-s1` | `spatial-v2` | 20 000 | phase2 240.8k (w/ cloze) | [bs62qs8x](https://wandb.ai/doodream/systemone-lite/runs/bs62qs8x) |

Full epoch ≈ 60.2k steps; 20k ≈ ⅓ epoch. Resume to 60.2k is supported via `--resume --max-steps 60200`.

## JevBench (local, T=1.0, 231 tasks)

| Model | Acc | ECE | p50 |
|---|---:|---:|---:|
| Qwen base | 39.4% | 0.284 | 13.6ms |
| mixed-sft (P1) | 45.9% | **0.221** | 13.5ms |
| spatial-v2 | 42.9% | 0.358 | 12.9ms |
| spatial-v2b | 43.7% | 0.306 | 13.7ms |
| **spatial-v2-s1** | **49.8%** | 0.307 | **12.6ms** |

Report: [`../benchmarks/jevbench_spatial_v2_s1.json`](../benchmarks/jevbench_spatial_v2_s1.json).

## Bare demos

[`../benchmarks/demos/spatial_v2_s1/`](../benchmarks/demos/spatial_v2_s1/) — chess / 2048 / gridworld / sokoban / connect4 GIFs (bare labels, no solver override).

## Strategy note (next)

Prioritize **S1-shaped** data (alerts / legal / recovery + stronger text share). Treat long-horizon optimal-move RL / self-play as a **sidekick** after an S1-optimized mix, not the mainline claim.

## Non-claims

- Not an official JevBench leaderboard submission.
- Held-out top-1 ≠ long rollout skill (demos still fail GridWorld/Sokoban clear).
- Hub mirrors: dataset `dwidlee/systemone-lite-phase2`, model `dwidlee/systemone-lite-0.5b` only.
