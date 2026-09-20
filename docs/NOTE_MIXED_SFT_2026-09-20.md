# Note: Mixed SFT with spatial chess state (2026-09-20)

What changed between the earlier general-only / FEN-era runs and the
`systemone-mixed-sft` run published as `dwidlee/systemone-lite-0.5b`.

Raw numbers: [`benchmarks/mixed_vs_base_report.json`](../benchmarks/mixed_vs_base_report.json).

---

## 1. Prior approach

### Training

| Track | Data | State encoding | Batching | Start |
|---|---|---|---|---|
| General-only SFT | ticket / alloc / debate (`general_train`, 32.4k) | Nested JSON text (readable) | Stratified by gym | Base 0.5B |
| Chess-only SFT | Stockfish distill (`chess_train_5k`, move rows) | **FEN string + metadata** | Chess tasks only | Base 0.5B |

General and chess were **separate checkpoints**. We treated chess as a
demo/ablation domain, not as part of the published general model.

### Evaluation

- **General:** iid + hard held-out (rule labels). Hard = layout / paraphrase /
  option-subset shift. Honest enough for that generator.
- **Chess:** move top-1 on a held-out JSONL. Criteria used letter aliases, but
  **option order was effectively stable** relative to how rows were built.
  Reported base ≈ 0.78–0.79 and chess-SFT ≈ 0.82–0.83 looked strong.

### What that implied (and where it misled)

1. **General-only SFT improved ticket/alloc** (iid ~0.68) but **hurt chess**
   when measured later under transfer (~0.46 on the old harness) — domain
   interference when chess was absent from the mix.
2. **FEN-only state** compressed an 8×8 board into one dense string. Same class
   of failure as injecting coordinate lists without a grid: a 0.5B LM does not
   reliably recover lines/diagonals from FEN alone.
3. **Chess accuracy was overstated** when option order correlated with labels
   (model bias toward early aliases). After shuffling, base falls to ~chance
   (~0.05 for ~20 legal moves).

---

## 2. This round

### Data

1. **Refresh chess JSONL** with `board_state()`:
   - `board_2d_map`: labeled 8×8 ASCII (files a–h, ranks 8–1)
   - keep FEN as secondary id
   - `gym: chess` in meta
2. **Keep** existing general train (already human-readable state).
3. **Merge** into `mixed_train.jsonl`:
   - 43 200 rows
   - equal gym caps: ticket / alloc / debate / chess = **10 800 each**
   - chess upsampled from 4 500 unique move rows

### Training

- **Cold start** from `Qwen/Qwen2.5-0.5B-Instruct` (do not continue from
  general-only or FEN-era chess weights).
- 1 epoch, batch size 4, `--stratified` (gym round-robin inside batches).
- 10 800 steps; equal sample counts per gym in the epoch.
- Checkpoint: `checkpoints/systemone-mixed-sft`.

### Evaluation (aligned with training)

| Bench | Protocol |
|---|---|
| General iid / hard | Same generators as before |
| Chess | `chess_eval_5k_2d.jsonl`: **2D map + shuffled** letter options |

Compare **base vs mixed** on all three. Old checkpoints on the new chess
harness are reference-only (they were not trained for this state format).

---

## 3. Results (base → mixed)

### General

| Split | Base | Mixed | Δ | Prior general-only |
|---|---:|---:|---:|---:|
| iid (n=3600) | 0.439 | **0.781** | +0.343 | 0.679 |
| hard (n=5400) | 0.427 | **0.733** | +0.307 | 0.652 |

Mixed beats both base and the earlier general-only SFT on the same splits.

### Chess (debiased 2D harness, n=500)

| Model | Accuracy |
|---|---:|
| Base | 0.050 |
| Mixed SFT | **0.236** |
| Prior general-only SFT | 0.042 |
| Prior chess-only (FEN-era) on this harness | 0.028 |

Chess rises with the mix instead of collapsing. Absolute level is still modest;
the gain is “above chance under an honest harness,” not engine strength.

---

## 4. Implications

1. **State must match the geometry of the task.**  
   For spatial domains, inject an explicit grid (`board_2d_map`), not only a
   compressed encoding (FEN). Text/JSON gyms were already fine; chess was not.

2. **Eval must not leak option-order bias.**  
   Shuffle aliases (or randomize criteria order) when scoring letter choices.
   Otherwise base “accuracy” can look like 0.8 while the policy is near-random
   under a fair shuffle.

3. **Multi-domain needs mixed batches from base, with balanced exposure.**  
   General-only specialization transferred poorly to chess. Equal-gym
   stratified mixing from a cold start improved **both** general and chess
   versus base, and improved general versus general-only SFT.

4. **Separate checkpoints are optional, not required for modest joint gains.**  
   One mixed head can hold ticket/alloc/debate/chess if the mix is balanced and
   state formats are correct. Domain tags / adapters remain available if chess
   must be pushed much higher without touching routing quality.

5. **Reported chess numbers before this note are not comparable** to the
   debiased 2D eval. Prefer `mixed_vs_base_report.json` and
   `chess_eval_5k_2d` going forward.

---

## 5. Reproduce

```bash
python scripts/chess_distill_dataset.py \
  --refresh-file data/chess_train_5k.jsonl \
  --out data/chess_train_5k_2d.jsonl

python scripts/build_mixed_distill.py \
  --general data/general_train.jsonl \
  --chess data/chess_train_5k_2d.jsonl \
  --out data/mixed_train.jsonl \
  --chess-target 10800

python scripts/chess_finetune.py \
  --data data/mixed_train.jsonl \
  --out checkpoints/systemone-mixed-sft \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --epochs 1 --batch-size 4 --tasks all --stratified
```

---

## 6. Open follow-ups

- More unique chess positions (less upsample) under the same 2D format  
- Restricted CE / DPO over best vs blunder aliases  
- Domain tags or LoRA if chess must climb without regressing debate  
- Reliability diagrams (option softmax ≠ population calibration)
