---
name: systemone-training-ops
description: >-
  Use this skill to inspect, resume, or execute training runs and LR pilots for systemone-lite,
  enforce postmortem guardrails (Issue #8-#12), generate synthetic diversity datasets (Issue #7),
  and run validation and benchmark evaluations.
---

# SystemOne Training Operations & Guardrails

Runbook and operational procedures for continuing `systemone-lite` model training, completing LR pilots, building synthetic diversity datasets, and running evaluation benchmarks.

---

## 1. Pre-flight Readiness Check

Before launching any GPU training or evaluation job:

```bash
# Run readiness checker from repo root
.agents/skills/systemone-training-ops/scripts/check_training_readiness.sh
```

**Key verification points**:
- **GPU VRAM**: Minimum 8,000 MiB free VRAM required. Check if background workloads (e.g. ComfyUI, web servers) need pausing.
- **Resume Guards**: Verify `pytest tests/test_finetune_resume_guards.py -q` passes before starting any training run.
- **Reference**: Review [postmortem_rules.md](./references/postmortem_rules.md) for invariants regarding scheduler horizon and LR jump prevention.

---

## 1.1 Live Training Monitoring

While training or LR pilots are running in background:

```bash
# Quick snapshot of active PID, GPU util/temp, latest steps, loss/EMA/lr/gnorm, and new checkpoints
.agents/skills/systemone-training-ops/scripts/monitor_training.sh
```

Or view real-time log stream:
```bash
tail -f logs/lr_pilots_qwen.log
```


---

## 2. Procedure: Complete LR Pilots (#12)

The pilots determine the optimal learning rate (`2e-6`, `5e-6`, or `1e-5`) starting from base `Qwen/Qwen2.5-0.5B-Instruct` with `batch_size=4`, `grad_accum=4`, cosine decay (10,000 max steps, 500 warmup).

### Step 2.1: Execute Pilots
```bash
# Automatically resumes Trial-A if checkpoints/lr-pilot-A-qwen-2e6/last exists, then runs B & C
bash scripts/run_lr_pilots.sh 2>&1 | tee logs/lr_pilots_qwen.log
```

### Step 2.2: Evaluate & Select Winning LR
Examine validation accuracy and loss trends across checkpoints (2,500 / 5,000 / 7,500 / 10,000 steps):

```bash
# Inspect best_val metrics across trials
for d in checkpoints/lr-pilot-*/best_val; do
    echo "=== $d ==="
    cat "$d/step.txt" 2>/dev/null || true
done
```

Selection criteria:
- Highest macro validation accuracy across tasks.
- Stable EMA loss curve without late-epoch divergence.
- Well-behaved gradient norm (`gnorm < 100`).

---

## 3. Procedure: Synthetic Diversity Expansion (#7)

Once the winning LR is established, expand training data diversity across Cellular Automata, Word Games, and NLP Cloze.

### Step 3.1: Generate Synthetic Data
```bash
# Generates 12k CA + 12k Word Games + 12k NLP Cloze
python scripts/build_synth_diversity.py \
    --out-dir data/synth_diversity \
    --seed 42
```

### Step 3.2: Contamination & Overlap Audit
```bash
python scripts/audit_train_eval_overlap.py \
    --train data/phase2_train_200k.jsonl \
    --eval data/phase2_eval_4k.jsonl
```

### Step 3.3: Launch SFT with Winning LR
Ensure the horizon is set at step 0:
```bash
python -u scripts/chess_finetune.py \
    --data data/phase2_train_synth_merged.jsonl \
    --out checkpoints/systemone-spatial-v2-diversity \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --epochs 1 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr <WINNING_LR> \
    --max-length 768 \
    --tasks all \
    --stratified \
    --max-steps 60000 \
    --warmup-steps 1000 \
    --scheduler cosine \
    --seed 7 \
    --save-every 5000 \
    --keep-checkpoints 3 \
    --eval-every 5000 \
    --eval-limit 800 \
    --wandb
```

---

## 4. Procedure: Benchmark & Release Gate

After training completion, evaluate decision quality, latency, and viral rollouts:

### Step 4.1: JevBench Evaluation
```bash
python scripts/jevbench_eval.py \
    --model checkpoints/<RUN_NAME>/best_val \
    --out benchmarks/jevbench_<RUN_NAME>.json
```

### Step 4.2: Bare Demos & Rollout Visualizations
```bash
python scripts/run_bare_demos.py \
    --model checkpoints/<RUN_NAME>/best_val \
    --out-dir benchmarks/demos/<RUN_NAME>
```

---

## 5. Troubleshooting & Guard Violations

- **`ResumeGuardError: LR jump detected`**:
  Do NOT bypass this error. It indicates the scheduler parameters or `--max-steps` differ from the run that created the checkpoint. Align `--max-steps` and `--lr` with the checkpoint or launch as a clean new run.
- **CUDA OOM on RTX 3060**:
  Ensure `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set and batch size is 4 with gradient accumulation.
