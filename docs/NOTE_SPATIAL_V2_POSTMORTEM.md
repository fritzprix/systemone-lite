# Postmortem: `systemone-spatial-v2` (Phase 2 gate checkpoint)

Date: 2026-09-21  
Scope: **completed** cold-start run `checkpoints/systemone-spatial-v2` (51 200 steps).  
**Not in scope:** `systemone-spatial-v2b` continual run — **still training** as of this note; no GPU work was done for this write-up.

Raw gate numbers: [`benchmarks/spatial_v2_report.json`](../benchmarks/spatial_v2_report.json).  
Bare demos (illustration only): [`benchmarks/demos/README.md`](../benchmarks/demos/README.md).  
Roadmap: [`docs/ROADMAP.md`](ROADMAP.md) · next synth work [#7](https://github.com/fritzprix/systemone-lite/issues/7).

---

## 1. Verdict

| Claim | Result |
|---|---|
| Phase 2 exit gate | **PASS** (`gate_pass: true`) |
| Spatial skill vs Phase 1 | Clear lift (overall 0.20 → **0.53**) |
| Long-horizon play quality | **Not** claimed — short demos mostly unsolved |
| Data hygiene at train time | **Mixed** — gate used pre-fix distill; pathologies found *after* |

**One line:** Alias-CE on bare spatial maps works for held-out top-1; GridWorld and chess are the weak spots; synth bugs inflated some scores until audited; v2b / remapped data are unfinished follow-ons.

---

## 2. What we trained

| Item | Value |
|---|---|
| Checkpoint | `checkpoints/systemone-spatial-v2` (`last` → `step-51200`) |
| Init | Cold start from `Qwen/Qwen2.5-0.5B-Instruct` (not Phase 1 continue) |
| Data | `data/phase2_train_200k.jsonl` (~204.8k), stratified |
| Steps / batch / max_len | 51 200 / 4 / 768 |
| Final avg loss | ~1.08 |
| Caps (approx) | spatial gyms 35k each; chess 32.4k; general 10.8k × 3 |

Trainer: `scripts/chess_finetune.py` (option-constrained CE on alias token).  
Mid-run infra added later (resume, `keep=3`, W&B) — **v2 itself** finished without W&B id in `train_meta.json`.

---

## 3. Gate results (honest harness)

Protocol: bare criteria, alias shuffle, no solver override. Suites in `benchmarks/_phase2_gate/`.

### Spatial held-out (n=500 / gym)

| Gym | Base | Phase 1 | **v2** | Δ vs P1 |
|---|---:|---:|---:|---:|
| Connect4 | 0.17 | 0.16 | **0.75** | +0.60 |
| Sokoban | 0.34 | 0.17 | **0.51** | +0.33 |
| 2048 | 0.31 | 0.19 | **0.50** | +0.31 |
| GridWorld | 0.32 | 0.27 | **0.36** | +0.09 |
| **Overall** | 0.28 | 0.20 | **0.53** | +0.33 |

### Other gates

| Gate | v2 | Rule | Pass? |
|---|---|---|---|
| Chess 2D shuffled | 0.22 | ≥ published P1 0.24 − 0.03 | yes (tight) |
| General iid / hard | 0.91 / 0.87 | no catastrophic forget vs P1 | yes (improved) |
| Invariance vs canonical | 0.55 vs 0.53 | within −0.08 | yes |

**Note:** Phase 1 chess **re-run** on the same harness scored 0.09 vs published ~0.24 — treat published chess as soft; v2’s 0.22 clears the written tolerance but is not a strong chess claim.

---

## 4. What worked

1. **Mixed spatial + chess + general** from base avoided destroying routing; general rose with more data/exposure.
2. **Bare labels + shuffle** kept the claim surface honest (no coaching keywords / no solver in the loop).
3. **Connect4 / 2048 / Sokoban** showed large top-1 lifts — short-horizon imitation is learnable at 0.5B.
4. **Invariance** tracked held-out — D₄/mirror (where applied) did not obviously collapse.
5. **Ops lessons locked in:** checkpoint rotation, resume, sparse val (~5k), “ask before long GPU” — after an unsolicited retrain was killed.

---

## 5. What hurt / what we learned too late

### 5.1 Synth pathologies (post-gate audit)

Gate checkpoint was trained on distill that later failed deeper audits. Fixes landed **after** v2 finished; HF `dwidlee/systemone-lite-phase2` and on-disk JSONL were rebuilt. **v2 weights do not include those fixes.**

| Issue | Effect |
|---|---|
| GridWorld fixed corner spawn | Pattern memorization; weak true navigation → small GW lift |
| Sokoban tiny template pool | High duplicate rate; trajectory memorization |
| Alert tasks class collapse (C4 / 2048; earlier Sokoban) | Easy yes/no hacking |
| 2048 `empty_cells_count` in state | Label leak for overflow alerts |
| Connect4 column bias | Distorted drop distribution |

After fixes: `AUDIT_PASS` (balanced alerts, no empty_count leak, higher C4 uniqueness).  
**Implication:** some of Connect4/alert-ish strength on the **old** eval may be partly distribution fit. Re-eval on fixed held-out is required before trusting v2 numbers as final (deferred while v2b occupies GPU).

### 5.2 Capability gaps (even if metrics look OK)

- **GridWorld** remains the weakest spatial gym (+0.09 vs P1).
- **Chess** barely gate-passes; unique positions still ~4.5k upsampled.
- **Bare demos:** 2048 improves (score 60 / tile 16); GridWorld/Sokoban/Connect4 **not cleared** in short clips — held-out ≠ rollout skill.
- Alert vs action mix: a large alert share can pad accuracy without teaching dynamics.

### 5.3 Process

- Full retrain started without ask → stopped; rule: no unsolicited long GPU jobs.
- First v2 train lacked mid-checkpoints / W&B; resume path was bolted on for **v2b**.
- Symbol remapping (~40%) rebuilt into JSONL **after** v2; current **v2b** still sees the **in-memory pre-remap** mix for this run.

---

## 6. Follow-on: `spatial-v2b` (finished — gate FAIL)

| Item | Value |
|---|---|
| Checkpoint | `checkpoints/systemone-spatial-v2b` (`step-20000`, W&B `s50jkm4d`) |
| Recipe | **Cold start from base**, `resume=False`, **20k steps only** (not continue from v2) |
| Gate report | [`benchmarks/spatial_v2b_report.json`](../benchmarks/spatial_v2b_report.json) → **`gate_pass: false`** |
| Demos | [`benchmarks/demos/spatial_v2b/`](../benchmarks/demos/spatial_v2b/) |

On the **current** (post-fix / symbol-remap) `phase2_eval_4k`:

| Model | Spatial overall | C4 | 2048 | GW | Sokoban |
|---|---:|---:|---:|---:|---:|
| Phase 1 re-run | 0.320 | 0.282 | 0.284 | 0.386 | 0.326 |
| **v2** (51.2k) re-run | **0.346** | 0.408 | 0.260 | 0.396 | 0.320 |
| **v2b** (20k from base) | 0.329 | 0.406 | 0.226 | 0.374 | 0.308 |

v2b also: chess **0.076**, general iid/hard 0.85 / 0.78 (forget OK). Failed gates: spatial lift, chess. Passed: general, invariance.

**Important:** Original v2 gate (0.53 spatial) was on the **pre-fix** eval. On the remapped/fixed eval, even full v2 only edges Phase 1 — symbol OOD + data fixes moved the goalposts. v2b does not beat v2.

**Lesson:** v2b was under-trained (20k vs 51k) and never a continual tune of the gate winner. Next: `--resume` from v2 on audited remapped JSONL (or full retrain), then re-gate on this eval.

---

## 7. Recommendations (ordered)

1. Keep **`systemone-spatial-v2`** as the Phase 2 reference until a real continue/retrain beats it on the fixed eval.
2. Next train: resume from v2 on audited + remapped JSONL (or full epoch), with mid-eval — not another 20k-from-base.
3. Treat **GridWorld + symbol remap + action-heavy mix** as the next data lever (feeds [#7](https://github.com/fritzprix/systemone-lite/issues/7)).
4. Do not start Phase 3 DPO until gate is re-confirmed on clean data with a checkpoint that beats v2.
5. Chess: more unique 2D positions or continue from Phase 1 weights if chess floor matters more than cold-start purity.

---

## 8. Artifact index

| Artifact | Role |
|---|---|
| `checkpoints/systemone-spatial-v2/` | Gate PASS weights (this postmortem) |
| `checkpoints/systemone-spatial-v2b/` | Ongoing continual — incomplete |
| `benchmarks/spatial_v2_report.json` | Gate JSON |
| `benchmarks/_phase2_gate/*__spatial_v2.json` | Per-suite dumps |
| `benchmarks/demos/spatial_v2/` | Bare GIFs (non-claims) |
| `scripts/audit_phase2_distill.py` | Post-hoc data hygiene |
| `src/systemone_lite/synth/symbol_aug.py` | Remap (post-v2 data) |
