#!/usr/bin/env bash
# LR pilots from Qwen2.5-0.5B-Instruct (#12).
# A resumes from step-2500 if present (OOM recovery); B/C start fresh.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
INIT=Qwen/Qwen2.5-0.5B-Instruct
DATA=data/phase2_train_200k.jsonl
mkdir -p logs checkpoints

run_trial() {
  local name="$1" lr="$2" out="$3"
  local extra=()
  if [[ -f "$out/last/train_state.pt" ]]; then
    extra+=(--resume)
    echo "==== $(date -Iseconds) RESUME $name from $out/last ===="
  else
    echo "==== $(date -Iseconds) START $name lr=$lr init=$INIT ===="
  fi
  python -u scripts/chess_finetune.py \
    --data "$DATA" \
    --out "$out" \
    --model "$INIT" \
    --epochs 1 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr "$lr" \
    --max-length 768 \
    --tasks all \
    --stratified \
    --max-steps 10000 \
    --warmup-steps 500 \
    --scheduler cosine \
    --seed 7 \
    --save-every 2500 \
    --keep-checkpoints 2 \
    --eval-every 2500 \
    --eval-limit 800 \
    --wandb \
    --wandb-project systemone-lite \
    --wandb-run-name "$name" \
    "${extra[@]}"
  echo "==== $(date -Iseconds) DONE $name ===="
}

run_trial trial-A-qwen-lr-2e6 2e-6 checkpoints/lr-pilot-A-qwen-2e6
run_trial trial-B-qwen-lr-5e6 5e-6 checkpoints/lr-pilot-B-qwen-5e6
run_trial trial-C-qwen-lr-1e5 1e-5 checkpoints/lr-pilot-C-qwen-1e5
echo "==== ALL PILOTS DONE $(date -Iseconds) ===="
python scripts/select_best_pilot.py

