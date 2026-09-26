# Training Operations & Resume Guardrails (Postmortem Rules)

Reference extracted from `NOTE_S1_LR_SCHEDULE_POSTMORTEM.md` (Issue #8, #9, #10, #11, #12).

---

## 1. Core Postmortem Invariants

1. **Scheduler Horizon Invariance**:
   - `total_steps` / `num_training_steps` must be fixed at step 0 and kept identical across all `--resume` sessions.
   - Never change `--max-steps` to extend a run whose scheduler has already decayed or finished.
   - If extending the horizon or changing peak LR is necessary, start a **NEW W&B run** and treat the previous checkpoint as weight initialization only.

2. **Resume Continuity & LR Jump Guard**:
   - `chess_finetune.py` validates that `|current_lr - last_saved_lr| / last_saved_lr <= 0.05`.
   - Any deviation above 5% raises a validation error preventing accidental restart shock.

3. **Checkpoint Retention Policy**:
   - `best_val` checkpoint must be saved to a dedicated directory (`best_val/`), exempt from `--keep-checkpoints` rotating cleanup.
   - Checkpoint symlink `last` must accurately point to the latest completed step.

4. **Metrics Integrity**:
   - Do NOT judge recovery or convergence solely by cumulative average loss (`train/avg_loss = running / step`).
   - Rely on:
     - `train/loss` (raw step loss)
     - `train/ema_loss` (exponential moving average)
     - `train/grad_norm`
     - Per-gym / stratified held-out validation accuracy.

---

## 2. Checkpoint Provenance

| Artifact / Checkpoint | Role | Rule |
|---|---|---|
| `checkpoints/systemone-spatial-v2-s1-pre-resume` | Safe baseline | Base weights for experiments; published to `dwidlee/systemone-lite-0.5b` |
| `checkpoints/systemone-spatial-v2-s1/step-50k..52k` | Contaminated schedule | **DO NOT USE** for downstream training decisions |
| `checkpoints/lr-pilot-A-qwen-2e6/last` | Pilot A in progress (~5.3k) | Resume directly to 10k using `--resume` |
