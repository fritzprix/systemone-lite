#!/usr/bin/env bash
# Pre-flight readiness check for systemone-lite training & pilots.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== [1/5] Checking GPU & VRAM ==="
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA device unavailable."
    exit 1
fi

TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
USED_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
FREE_MEM=$((TOTAL_MEM - USED_MEM))
echo "Total VRAM: ${TOTAL_MEM} MiB | Used: ${USED_MEM} MiB | Free: ${FREE_MEM} MiB"

if [ "$FREE_MEM" -lt 8000 ]; then
    echo "WARNING: Free VRAM is under 8000 MiB (current: ${FREE_MEM} MiB)."
    echo "Processes currently using GPU:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    echo ">> Please free GPU memory before launching training pilots to avoid OOM."
else
    echo "OK: Sufficient VRAM available."
fi

echo ""
echo "=== [2/5] Checking Dataset Assets ==="
for f in data/phase2_train_200k.jsonl data/phase2_eval_4k.jsonl; do
    if [ -f "$f" ]; then
        echo "OK: Found $f ($(du -h "$f" | cut -f1))"
    else
        echo "ERROR: Missing dataset $f"
    fi
done

echo ""
echo "=== [3/5] Checking LR Pilot Checkpoints ==="
PILOT_A_DIR="checkpoints/lr-pilot-A-qwen-2e6"
if [ -d "$PILOT_A_DIR" ]; then
    echo "Found Pilot A checkpoint dir: $PILOT_A_DIR"
    if [ -L "$PILOT_A_DIR/last" ]; then
        TARGET=$(readlink "$PILOT_A_DIR/last")
        echo "OK: $PILOT_A_DIR/last points to $TARGET"
        if [ -f "$PILOT_A_DIR/last/train_state.pt" ]; then
            echo "OK: $PILOT_A_DIR/last/train_state.pt exists (ready to --resume)"
        else
            echo "WARNING: train_state.pt missing in $PILOT_A_DIR/last"
        fi
    else
        echo "NOTE: $PILOT_A_DIR/last symlink not found."
    fi
else
    echo "NOTE: Pilot A dir not found (fresh start)."
fi

echo ""
echo "=== [4/5] Checking Resume Guards Test ==="
if pytest tests/test_finetune_resume_guards.py -q; then
    echo "OK: Resume guards tests passed."
else
    echo "ERROR: Resume guard tests failed. Fix before running training."
    exit 1
fi

echo ""
echo "=== [5/5] Pre-flight Summary ==="
if [ "$FREE_MEM" -ge 8000 ]; then
    echo "READY: Environment is clear to run/resume training."
else
    echo "BLOCKED: Free GPU memory required before running training."
fi
