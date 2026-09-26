#!/usr/bin/env bash
# Quick training monitor for systemone-lite
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "    SystemOne Training Live Monitor       "
echo "=========================================="
echo "Timestamp: $(date -Iseconds)"

echo ""
echo "--- 1. Training Processes ---"
RUNNING_PID=$(pgrep -f "chess_finetune.py" || true)
if [ -n "$RUNNING_PID" ]; then
    echo "STATUS: Active training detected (PID: $RUNNING_PID)"
    ps -o pid,user,%cpu,%mem,etime,cmd -p "$RUNNING_PID"
else
    echo "STATUS: No active chess_finetune training process found."
fi

echo ""
echo "--- 2. GPU Utilization ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader \
        | awk -F, '{printf "GPU Util: %s | VRAM: %s / %s | Temp: %s | Power: %s\n", $1, $2, $3, $4, $5}'
else
    echo "nvidia-smi unavailable."
fi

echo ""
echo "--- 3. Latest Training Metrics (from active log) ---"
LATEST_LOG=$(ls -t logs/*.log 2>/dev/null | grep -E "lr_pilots|train" | head -n 1 || true)
if [ -n "$LATEST_LOG" ]; then
    echo "Log file: $LATEST_LOG"
    echo "Recent training steps:"
    grep -E "step=[0-9]+" "$LATEST_LOG" | tail -n 5 || echo "No step lines found yet."
    echo ""
    echo "Recent validation evaluations:"
    grep -E "val/acc|val accuracy|best_val" "$LATEST_LOG" | tail -n 3 || echo "No recent val logs."
else
    echo "No recent training log found in logs/."
fi

echo ""
echo "--- 4. Latest Checkpoints ---"
LATEST_CKPT=$(ls -td checkpoints/*/* 2>/dev/null | head -n 5 || true)
if [ -n "$LATEST_CKPT" ]; then
    echo "$LATEST_CKPT"
else
    echo "No checkpoints found."
fi
echo "=========================================="
