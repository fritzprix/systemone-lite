# Training Metrics & Health Diagnostic Guidelines

Reference manual for interpreting SystemOne-lite training metrics (local logs & W&B).

---

## 0. Anti-Spin & Skeptical Assessment Invariants (철칙)

* **금지 표현**: "모범적인", "안정적으로 수렴", "신기록", "🎉" 등 근거 없는 낙관 수식어 전면 금지.
* **표본 오차 (`n=800`)**: ±1.5%p 미만의 수치 변동은 개선이 아닌 **통계적 노이즈 / 정체(Stagnation)**로 처리. 단 몇 문제 더 맞힌 것을 '성능 개선'으로 포장 금지.
* **계단식 Loss 시프트**: Loss 베이스라인이 0.2 이상 점프하면 gnorm 수치와 상관없이 즉시 **데이터셋 소진/왜곡 결함**으로 에스컬레이션.
* **학습 효용 소진 판정**: LR이 피크의 10% 이하로 떨어진 상태에서 수천 스텝 동안 val 정체 시 **미련 없이 '실패' 판정 및 즉시 조기 종료/차기 Trial 전환 직언**.

---

## 1. Primary Metrics Reference

| Metric | Source | Normal Range | Warning / Anomaly Condition |
|---|---|---|---|
| `train/loss_ema_100` | W&B / Log | 0.8 ~ 1.5 | 계단식 급등 (+0.2 이상), 500 스텝 연속 상승, 또는 > 2.0 |
| `train/gradient_norm` | W&B / Log | 40 ~ 100 | > 150 (Gradient explosion / clipping saturated) |
| `train/lr` | W&B / Log | Follows cosine curve | Discontinuity > 5% on resume, 또는 피크의 10% 이하에서 성능 정체 시 효용 소진 |
| `val/accuracy` | W&B / Checkpoint | 45% ~ 60% (stratified) | ±1.5%p 이내 변동은 '정체(Noise)'로 처리, 연속 하락 시 즉시 경보 |
| `train/avg_loss` | W&B / Log | Informational only | **Do not** use cumulative average to evaluate current convergence |

---

## 2. Gym Breakdown Expectations

| Gym Family | Target Val Accuracy | Characteristics |
|---|---|---|
| **NLP Cloze / Ticket** | **80% ~ 95%** | Strong natural language anchor tasks; should converge first. |
| **Word Games / Debate** | **50% ~ 65%** | Closed-choice rule-following. |
| **Cellular Automata** | **50% ~ 60%** | Local state transition prediction. |
| **GridWorld / 2048 / Sokoban** | **25% ~ 45%** | Spatial heuristic planning. |
| **Chess / Connect4** | **10% ~ 30%** | Multi-step combinatorial search space; gradual improvement. |

---

## 3. Intervention Playbook

1. **If `gradient_norm` spikes > 150 repeatedly**:
   - Check if learning rate is too aggressive or batch size / accum factor is too small.
2. **If `loss_ema` diverges upward**:
   - Stop run immediately. Revert to previous checkpoint and halve the peak learning rate.
3. **If GPU VRAM free drops < 800 MiB**:
   - Check background processes with `nvidia-smi` and reduce `--batch-size` or set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
