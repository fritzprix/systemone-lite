# Dataset Audit Report — systemone-lite (RE-VISITED & CORRECTED)

**Date:** 2025-06-30  
**Scope:** All `.jsonl` files under `data/` (23 files, **452,720 lines**, ~563 MB total)  
**Status:** Full re-verification against user's detailed feedback

---

## 1. Inventory (CORRECTED)

| File | Lines | Size | Tasks | Gyms |
|------|------:|-----:|------:|-----:|
| `chess_distill.jsonl` | 360 | 410K | 120 | 3 | Parent pool |
| `chess_distill_5k.jsonl` | 5,000 | 7.1M | 5,000 | 1 | Parent pool |
| `chess_eval.jsonl` | 54 | 64K | 20 | 3 | |
| `chess_eval_5k.jsonl` | 500 | 724K | 500 | 1 | |
| `chess_eval_5k_2d.jsonl` | 500 | 664K | 500 | 1 | |
| `chess_train.jsonl` | 306 | 348K | 104 | 3 | |
| `chess_train_5k.jsonl` | 4,500 | 6.4M | 4,500 | 1 | |
| `chess_train_5k_2d.jsonl` | 4,500 | 5.8M | 4,500 | 1 | |
| `general_distill.jsonl` | 36,000 | 30M | 4,000 | 3 | Train pool only |
| `general_eval.jsonl` | 3,600 | 3.0M | 400 | 3 | |
| `general_eval_hard.jsonl` | 5,400 | 4.5M | 600 | 3 | |
| `general_train.jsonl` | 32,400 | 27M | 3,600 | 3 | |
| `mixed_train.jsonl` | 43,200 | 42M | 10,800 | 4 | |
| `phase2_eval_4k.jsonl` | 4,300 | 3.5M | 496 | 10 | |
| `phase2_eval_invariance.jsonl` | 2,300 | 2.1M | 500 | 8 | |
| `phase2_smoke_10k.jsonl` | 10,000 | 9.2M | 3,000 | 8 | |
| `phase2_train.jsonl` | 32,400 | 33M | 10,800 | 8 | Legacy |
| `phase2_train_200k.jsonl` | 228,800 | 183M | 33,996 | 10 | **Main** |
| `spatial_eval_1k.jsonl` | 1,000 | 772K | 164 | 4 | |
| `spatial_train_10k.jsonl` | 10,800 | 8.2M | 1,812 | 4 | |
| `spatial_train_2k.jsonl` | 2,000 | 1.6M | 340 | 4 | |
| `synth_diversity_eval.jsonl` | 800 | 512K | 260 | 2 | |
| `synth_diversity_train.jsonl` | 24,000 | 16M | 4,791 | 2 | |

**Grand total: 452,720 lines across 23 files, ~563 MB**

---

## 2. Schema Metadata Distribution (CORRECTED)

| Status | Count | % |
|--------|------:|--:|
| Has `schema` key | 379,000 | 83.7% |
| **Missing `schema` key** | **73,720** | **16.3%** |

**Gyms missing `schema` key:**
| Gym | Entries |
|-----|--------:|
| chess | 63,000 |
| move (chess) | 10,240 |
| piece (chess) | 240 |
| destination (chess) | 240 |

**All chess entries lack `schema` metadata.** This is likely by design (chess uses `move`, `piece`, `destination` tasks without the standard schema classification), but it represents a structural inconsistency.

---

## 3. Verified Findings

### 3.1 ✅ Data Leakage — General Datasets (Confirmed, 30–38%)

`general_train` ∩ `general_eval` = **192 exact state matches** (sample 5000 ∩ 500).

| Train Set | Eval Set | Matches | Leak Rate |
|-----------|----------|--------:|----------:|
| `general_train` | `general_eval` | 192 | ~38% of eval sample |
| `general_train` | `general_eval_hard` | 20 | ~4% of hard eval sample |

**Status:** This is a confirmed, pre-existing issue. The `general_build_meta.json` specifies `eval_frac: 0.1` but actual overlap is far higher.

### 3.2 ✅ Phase 2 Leakage — Already Fixed

| Comparison | Leak Rate |
|------------|----------:|
| `phase2_train_200k` (main) vs `phase2_eval_4k` | **0.00%** ✅ |
| `phase2_train` (legacy 32k) vs `phase2_eval_4k` | 1.53% |

**Status:** The user's prior patch has already resolved the main dataset. The legacy `phase2_train.jsonl` still has minor leakage but it is not the primary training set.

### 3.3 ✅ Chess Distill Files Are Parent Pools (Not Duplicates)

Verified: `chess_distill.jsonl` states == `chess_train.jsonl` states ∪ `chess_eval.jsonl` states (196 = 178 + 49, exact set equality).

**Structure:**
| File | Role |
|------|------|
| `*_distill.jsonl` | Parent pool (train + eval union) |
| `*_train.jsonl` | Training split |
| `*_eval*.jsonl` | Evaluation split |

**Distill files are NOT duplicates of train** — they are the original generation pool. The overlap is by design.

### 3.4 ✅ Label Alias Imbalance — Simpson's Paradox (Not a Real Issue)

The apparent A/B bias (39% each) in `general_train` is entirely explained by the high proportion of 2-choice tasks:

- 60%+ of entries are 2-choice tasks (debate.winner, debate.enough_evidence, ticket.needs_human, etc.)
- Within 2-choice tasks: A = 49.6%, B = 50.4% — perfectly balanced
- Multi-choice tasks also have shuffled alias assignment

**Verdict:** No real bias. The overall distribution is an aggregation artifact.

---

## 4. 🚨 Blind Spots Found by User — Confirmed Critical Issues

### 4.1 Coaching Keyword Leakage in Criteria — CRITICAL

Chess heuristics leaked directly into **criteria** text, giving away the answer:

| Keyword | phase2_train | chess_train | chess_train_5k |
|---------|:----------:|:-----------:|:-------------:|
| `controls center` | 9,900 | 90 | 4,112 |
| `develops minor` | 9,541 | 89 | 3,977 |
| `king safety` | 21,600 | 100 | 4,500 |
| `development` | 21,600 | 100 | 4,500 |
| `tactics` | 21,600 | 100 | 4,500 |

**Location:** These terms appear in the `criteria` field descriptions (e.g., `"O": "e7e5: to e5, controls center"`), effectively providing Chain-of-Thought hints that tell the model which option is preferred.

**Impact:** Model may learn to match keywords in criteria to correct labels rather than actually reasoning about the position.

### 4.2 `hazard_nearby` Field Leakage in State — HIGH

| Entry | Location | Count |
|-------|----------|------:|
| `hazard_nearby: true/false` | `state` field | 2,700 |

This pre-computed boolean directly answers the `gridworld.hazard_alert` task, bypassing any spatial reasoning.

### 4.3 `empty_cells_count` Leakage in 2048 — HIGH

| Field | Count |
|-------|------:|
| `empty_cells_count` | 3,450 |

This pre-computed field directly indicates overflow risk in 2048 tasks, giving away the answer to `game2048.overflow_alert`.

### 4.4 Alert Task Class Collapse — CRITICAL

| Task | Train Label Distribution | Severity |
|------|------------------------|----------|
| `sokoban.deadlock_alert` | **B: 100%** (A: 0%) | 🔴 Fatal |
| `connect4.threat_alert` | **B: 95%** (A: 5%) | 🔴 Fatal |
| `game2048.overflow_alert` | **B: 96%** (A: 4%) | 🔴 Fatal |
| `gridworld.hazard_alert` | **B: 70%** (A: 30%) | 🟡 Severe |

**Impact:** The model can achieve high accuracy by simply outputting the majority class without any reasoning. This is a fundamental data quality failure.

---

## 5. Recommendations

### Immediate (must fix before any training)

| Priority | Action |
|----------|--------|
| **P0** | Remove coaching keywords from chess criteria (`controls center`, `develops minor`, `king safety`, `development`, `tactics`) — 9,500–45,000 entries affected |
| **P0** | Remove `hazard_nearby` pre-computed field from gridworld state |
| **P0** | Remove `empty_cells_count` from 2048 state (or only use in non-alert tasks) |
| **P0** | Rebalance alert tasks — current distributions make classification trivial without reasoning |

### Short-term

| Priority | Action |
|----------|--------|
| **P1** | Regenerate `general_eval` with strict deduplication against `general_train` |
| **P1** | Add `schema` key to all non-chess entries for consistency |
| **P2** | Clean up duplicate states within files (24–30% in some files) |

### Long-term

| Priority | Action |
|----------|--------|
| **P2** | Add data integrity CI checks: keyword leakage, label balance, eval/train overlap |
| **P2** | Document distill/train/eval pipeline structure in README |

---

## 6. Corrections from Previous Version

| Claim (Previous) | Verdict | Correction |
|-----------------|---------|------------|
| ~530,870 total lines | ❌ Calculation error | **452,720 lines** (confirmed by `wc -l`) |
| 0% entries missing schema | ❌ Overlooked chess | **16.3% (73,720)** chess entries lack `schema` key |
| Phase 2 leakage = 0.4% | ❌ Wrong file sampled | Main file `phase2_train_200k` = **0.00%** (already fixed) |
| chess_distill is a duplicate | ❌ Pipeline misunderstanding | It is the **parent pool** (train ∪ eval) |
| A/B label bias is a real issue | ❌ Simpson's paradox | **No real bias** — 2-choice tasks are 50:50 |
| Coaching keywords: not detected | ❌ Major blind spot | **Confirmed massive leakage** in criteria |
| Alert task balance: ok | ❌ Major blind spot | **Class collapse: 95–100%** for most alert tasks |
