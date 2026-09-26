# Postmortem: `action-v2-qwen` (published weights, 2026-09-26)

Scope: finished run `checkpoints/systemone-action-v2-qwen` (W&B `dz7qvphm`).  
AGENTS.md: ±1.5%p on n=800 = noise; no optimistic spin.

Raw reports:

- Held-out full test: [`benchmarks/phase2_heldout__action_v2_qwen.json`](../benchmarks/phase2_heldout__action_v2_qwen.json)
- First-800 protocol: [`benchmarks/phase2_first800__action_v2_qwen.json`](../benchmarks/phase2_first800__action_v2_qwen.json)
- JevBench: [`benchmarks/jevbench_action_v2_qwen.json`](../benchmarks/jevbench_action_v2_qwen.json)

---

## 1. Verdict

| Claim | Result |
|---|---|
| Connect4 / GridWorld `action_v2` redesign | **Supported** (large lift vs staged-c-init on same LR) |
| Staged chess compounding through late LR | **Not supported** — mid-run peak then noise-regress |
| Overall first-800 “64.88% best” as clean series | **Caveat** — sokoban eval patched mid-run; step-5000 weights pruned |
| JevBench lift vs prior Hub weights | **Not claimed** — 50.65% vs 49.8% ≪ SE (~±3%p, n=231) |
| Publish readiness | **Yes** — full test + JevBench remeasured; dataset must ship with model |

**One line:** Legal-only spatial options help Connect4/GW; 2048 and late chess remain weak; mid-run sokoban eval skew broke the val series; publish on full-test + JevBench numbers, not the broken mid-run curve.

---

## 2. What we trained

| Item | Value |
|---|---|
| Checkpoint | `checkpoints/systemone-action-v2-qwen/best_val` (= step 10000) |
| Init | Cold `Qwen/Qwen2.5-0.5B-Instruct` |
| Data | `data/phase2_train_200k.jsonl` (240 800) — staged chess + action_v2 spatial + prior NLP/CA/word mix |
| LR / sched | 1e-5 cosine, warmup 500, max 10 000 |
| Batch | 4 × grad_accum 4, stratified |
| W&B | `doodream/systemone-lite/dz7qvphm` |
| Sampler | ~14.5k / gym — no late-domain exhaustion in counts |

Task changes vs prior Hub mix:

- **Chess `staged_v1`:** piece + destination (option caps ≤8), not full uncapped move lists.
- **Spatial `action_v2`:** legal-only action options; Connect4 drop≤3 + win_now; alerts retained/upweighted.
- **Sokoban eval (mid-run patch):** deadlock alerts rebalanced 75/75 yes/no; remap_alert_prob≈0.35; first-800 alerts ~12/11.

---

## 3. Train-time val (first 800) — series broken

| step | overall | chess | connect4 | sokoban | gridworld | 2048 | eval file |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2500 | 61.38% | 45.2 | 42.0 | 48.1 | 40.4 | 36.8 | **old** sokoban skew |
| 5000 | 63.88% | 50.5 | 51.0 | **35.1** | 52.8 | 38.2 | **old** |
| 7500 | 63.12% | 46.0 | 47.7 | 41.4 | 45.8 | 35.7 | **patched** |
| 10000 | 64.88% | 44.8 | 50.0 | 47.1 | 53.1 | 35.7 | **patched** |

- 2.5k→5k **+2.5%p**: above noise on a fixed eval.
- 5k→7.5k: **do not compare** (eval discontinuity).
- 7.5k→10k **+1.75%p**: ~1 SE — not a generalization claim.
- LR ≤ 10% of peak from step **8055**; post-low best Δ **+1.0%p** = noise → **late-run utility exhausted**. Same-run resume past LR=0 forbidden (#8).

**step-5000 weights were pruned** (`keep-checkpoints`); cannot re-score 5k on patched eval. Permanent hole.

Train health: no EMA step-jump ≥0.2; warmup gnorm spikes settled; late EMA mild rebound as LR→0.

---

## 4. Post-train honest metrics (publish surface)

Protocol: alias shuffle, seed 0, `scripts/phase2_eval.py` / `scripts/jevbench_eval.py`.

### Phase2 held-out (`test`, n=4700)

| | |
|---|---|
| Overall | **61.55%** (2893/4700) |
| First-800 re-eval | **63.88%** (vs train-logged 64.88% — shuffle/noise) |

| Gym | n | Acc |
|---|---:|---:|
| ticket_dungeon | 335 | 100.0% |
| resource_allocator | 339 | 98.2% |
| nlp_cloze | 400 | 89.0% |
| debate_judge | 326 | 86.8% |
| word_games | 400 | 86.8% |
| cellular_automata | 400 | 50.5% |
| connect4 | 500 | 47.0% |
| gridworld | 500 | 45.6% |
| chess | 500 | 42.0% |
| sokoban | 500 | 38.2% |
| game2048 | 500 | 34.6% |

Weak tasks: `connect4.drop` 34%, `game2048.slide` 34%, chess `piece` 36%, `sokoban.direction` 36%.  
Alert / ticket / alloc remain near ceiling — they pad aggregate.

### JevBench public (231, T=1.0)

| | This publish | Prior Hub (`spatial_v2_s1`) |
|---|---:|---:|
| Accuracy | **50.65%** | 49.78% |
| ECE | **0.245** | 0.307 |
| Latency p50 | 13.1 ms | 12.6 ms |

Δacc ≈ +0.9%p ≪ SE — **not an improvement claim**. ECE drop is directionally better but not over-interpreted here.

---

## 5. Comparison caveats

| Run | Final first-800 | Notes |
|---|---:|---|
| Trial C Qwen 1e-5 | 53.25% | older mix |
| staged-c-init | 57.38% | staged chess, pre-action_v2 spatial |
| action-v2-qwen | 64.88% train / 63.88% re-eval | new spatial + mid-run sokoban patch |

Absolute ranking across runs is **not** apples-to-apples (data + eval differ). Domain-level Connect4/GW lifts vs staged-c-init remain the strongest evidence for `action_v2`.

---

## 6. Publish checklist

- [x] Full `test` re-eval + JevBench on `best_val`
- [x] train∩test overlap audit PASS (0.00% state_task)
- [x] Hub dataset + model + cards updated together
- [ ] step-5000 on patched eval — **impossible** without retrain
- [ ] Formal `_phase2_gate` suite refresh — optional follow-up; not blocking this publish

---

## 7. Next (not this publish)

1. **game2048** task redesign (clear bottleneck on full test).
2. Keep more mid-run step checkpoints (or W&B artifacts) so eval patches do not orphan the series.
3. Chess late-run regression: staged alone is insufficient under cosine tail.
