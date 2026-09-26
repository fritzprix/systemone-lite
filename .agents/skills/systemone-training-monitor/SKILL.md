---
name: systemone-training-monitor
description: >-
  Use this skill when checking, diagnosing, or tracking intermediate training status for systemone-lite,
  inspecting W&B logs and summaries, verifying gradient/loss health, and monitoring active training runs.
---

# SystemOne Training Live Diagnostics & W&B Monitoring

This skill provides step-by-step procedures and tools for inspecting in-progress or paused `systemone-lite` training runs, analyzing real-time W&B logs and summaries, and detecting anomalies.

---

## 1. Quick Intermediate Health Check

To generate a full diagnostic report across processes, GPU, local logs, and W&B:

```bash
python .agents/skills/systemone-training-monitor/scripts/check_training_health.py
```

### What this checks automatically:
1. **Liveness**: Detects active PID (`chess_finetune.py`), CPU/MEM usage, elapsed runtime.
2. **GPU Hardware**: GPU utilization %, free VRAM margin, temperature (°C), power draw.
3. **W&B Live Telemetry**: Queries W&B API for run status, step count, Raw/EMA/Avg loss, current LR, and `gradient_norm`.
4. **Per-Task Accuracy**: Visual breakdown of validation accuracy across all gym families (`nlp_cloze`, `chess`, `connect4`, `cellular_automata`, etc.).
5. **Local Step Logs**: Inspects the latest 5 steps logged on disk.

---

## 2. Interpreting W&B Logs & Diagnostics

Refer to [metric_diagnostics.md](./references/metric_diagnostics.md) for detailed reference ranges.

### Key Indicators:
* **`train/loss_ema_100`**: Exponential moving average over 100 steps. If this trends steadily upward for > 500 steps, flag early divergence.
* **`train/gradient_norm`**: Should stay within 40–100. Values consistently > 150 indicate gradient clipping saturation and risk of destabilization.
* **`train/lr`**: Ensure learning rate follows the intended schedule curve without abrupt drops or jumps.
* **`val/accuracy` by Gym**:
  * Cloze / Ticket Dungeon should maintain **> 80%**.
  * Spatial games (GridWorld, Sokoban, 2048) typically score **25%–45%**.
  * Combinatorial games (Chess, Connect4) improve slowly (**10%–30%**).

---

## 3. Scheduled / Background Monitoring

When running overnight or long pilot runs, set up proactive checks:

### Periodic Check with `/schedule`
Request the agent to monitor training at fixed intervals:
- "Check training health every 15 minutes and report if loss EMA rises above 1.5 or GPU temperature exceeds 80°C."

### Real-Time Tail
```bash
# Follow local output
tail -f logs/lr_pilots_qwen.log

# Or inspect latest W&B output log
tail -f wandb/latest-run/files/output.log
```

---

## 4. Anomaly Response Playbook

- **Loss Spike or Divergence**: Stop the process (`kill <PID>`), inspect the last 50 steps in W&B, and resume from the previous checkpoint with halved LR.
- **OOM Warning**: Free background GPU processes or enable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Scheduler Discontinuity**: Ensure `--max-steps` matches the checkpoint horizon exactly to avoid restart shock.
