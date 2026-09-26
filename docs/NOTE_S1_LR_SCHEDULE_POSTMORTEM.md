# `bs62qs8x` 학습 포스트모템

**Run:** `spatial-v2-s1-cloze`  
**W&B:** https://wandb.ai/doodream/systemone-lite/runs/bs62qs8x  
**최종 상태:** `killed` — `KeyboardInterrupt`  
**종료 step:** 52,124 / 60,200  
**최종 val accuracy:** 54.25%  
**최고 val accuracy:** 56.00% @ 40k  
**판정:** 학습 발산 실패가 아니라 **실험 설계 + resume 연속성 실패로 결과 해석력이 훼손된 run**

**코드 근거:** `scripts/chess_finetune.py` (로컬 git; W&B에는 code artifact 없음)  
**메트릭 근거:** W&B history + `logs/phase2_s1_cloze_train*.log`  
**관련 이슈:** #8 (scheduler), #9 (best ckpt), #10 (metrics), #11 (artifacts), #12 (LR pilot)

---

## 1. Executive summary

이 run은 20k에서 끝난 학습을 60.2k까지 같은 W&B run으로 이어 붙였지만, resume 전후 learning rate가 연속적이지 않았다.

| | LR |
|---|---|
| 1차 구간 종료 (step 20 000, horizon=20 000) | **`0`** (W&B 샘플) / 직전 step은 ~0 |
| 재개 직후 (horizon=60 200으로 재구성) | **`≈1.48e-5`** |

따라서 하나의 연속 최적화 궤적이 아니라, 같은 run history 안에 **서로 다른 LR schedule을 가진 두 stage**가 연결된 형태다.

재개 후:

- 25–30k에서 training loss가 잠시 낮음
- 30k 이후 raw training loss 상승
- val은 대체로 53–54%대 정체
- 40k에서 56%가 나왔으나 45k 54%, 50k 54.25%로 **유지되지 않음**
- 52,124에서 사용자 중단

**40k checkpoint는 국소 최고점 후보일 뿐, 지속 일반화 개선의 증거는 아니다.**

---

## 2. 확정 사실 vs 추정

### 2.1 코드로 확정된 사실

`chess_finetune.py`는 scheduler를 이렇게 만든다:

```python
total_steps = epochs * steps_per_epoch
if max_steps is not None:
    total_steps = min(total_steps, max_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(1, total_steps // 10),
    num_training_steps=max(1, total_steps),
)
```

| 사실 | 근거 |
|---|---|
| 1차 CLI가 `--max-steps 20000` | `wandb/run-20260922_071204-bs62qs8x` args; log `steps≈20000` |
| 그 설정에서 step 20 000의 LR은 **정확히 0** | `get_linear_schedule_with_warmup(..., num_training_steps=20000)` 재현 |
| 2·3차 CLI가 `--max-steps 60200` + `--resume` | local wandb configs + continue/full logs |
| 동일 global step=20 000을 horizon=60 200으로 다시 계산하면 LR≈**1.484e-5** | 동일 API로 재현; W&B bootstrap 직후 LR과 일치 |
| checkpoint에 optimizer/scheduler/scaler/RNG는 저장됨 | `save_checkpoint` → `train_state.pt` |
| 그러나 resume 시 **새** scheduler를 `현재` `total_steps`로 만든 뒤 `load_state_dict` | LambdaLR의 `lr_lambdas`는 새 closure; horizon이 바뀌면 같은 `last_epoch`라도 LR이 달라짐 |
| `train/avg_loss`는 **누적 평균** | `running += loss; avg = running / step` |

로컬에서 horizon만 바꿔 재현한 값:

```text
horizon=20000 step=20000  → lr = 0.0
horizon=60200 step=20000  → lr ≈ 1.484e-5
```

이슈 #8의 `7.78e-8 → 1.48e-5 (~190×)`는 **같은 메커니즘**의 근사 샘플(직전 step / 로깅 시점 차이)로 보면 된다. W&B 5k 간격 샘플에서는 `0 → 1.30e-5`(step 20k→25k)로 보인다.

### 2.2 로그·W&B로 확정된 사실

| 사실 | 근거 |
|---|---|
| step 20 000 val 53.63%, avg_loss ≈ 0.887, lr=0 | W&B |
| resume 후 같은 run id `bs62qs8x`에 history 연결 | W&B + logs |
| 40k val **56.00%** (448/800), 이후 45k 54%, 50k 54.25% | W&B |
| 35k→40k 순증가 **+22** 정답; 구성은 아래 §3 | W&B rates × eval 앞 800행 gym 분포 |
| 52,124에서 KeyboardInterrupt / `killed` | W&B state + local log |
| `batch_size=4`, grad accum 없음 (config) | W&B config |
| `val/n=800`, `eval` = 파일 앞 800행 (비층화) | `run_validation` + `phase2_eval_4k.jsonl` |
| `keep_checkpoints=3` | CLI / config |
| W&B에 training code / dataset / ckpt artifact 없음 | run files = config, output.log, requirements, metadata |

### 2.3 추정 (개연성 높음, 코드 한 줄로 “당시 프로세스 내부”까지는 미증명)

| 추정 | 메모 |
|---|---|
| 작은 physical batch가 높은 LR 재시작 충격을 키움 | 기여 요인; 인과 단일 원인 아님 |
| 40k 56%가 noise/국소 변동 | paired prediction 미저장 → McNemar 불가; 45·50k 비재현이 강한 정황 |
| step-40k 가중치가 `keep=3`에 의해 삭제됨 | 정책상 개연성 큼; 현재 디스크에 step-50/51/52k만 잔존 |

### 2.4 W&B만으로는 확정 못 했던 것 → 로컬 코드로 보완

W&B tracked files에 `chess_finetune.py`가 없어 “스케줄러를 어떻게 만들었는지”는 run artifact만으로는 증명 불가였다.  
**로컬 스크립트 + CLI args + LR 수치 재현**으로 위 §2.1을 **확정**한다.

---

## 3. 타임라인

| 구간 | 관찰 |
|---|---|
| 0–20k | warmup 후 linear decay → **LR=0 @ 20k** (horizon=20k) |
| ~20k | 장시간 중단 |
| 20k+ | `--max-steps 60200` resume / bootstrap → LR≈1.48e-5 (horizon=60.2k) |
| 25–30k | `train/avg_loss` 최저대 (~0.874–0.878) |
| 30–35k | raw loss 상승, val 53.25% |
| 40k | val **56.00%** (단일 최고) |
| 45k | val 54.00% |
| 50k | val 54.25% |
| 52k | checkpoint |
| 52,124 | `KeyboardInterrupt`, `killed` |

---

## 4. 결과 요약

### Validation accuracy

| Step | Accuracy |
|---:|---:|
| 5k | 54.50% |
| 10k | 53.25% |
| 15k | 53.25% |
| 20k | 53.63% |
| 25k | 54.38% |
| 30k | 54.25% |
| 35k | 53.25% |
| 40k | **56.00%** |
| 45k | 54.00% |
| 50k | **54.25%** |

Resume 이후 중앙값 ≈ **54.25%**. 40k 56%는 장기 추세가 아니라 단일 checkpoint 국소 최고점.

### 40k spike — task별 정답 수 (확정)

Eval = `phase2_eval_4k.jsonl` **앞 800행** (gym 고정).  
35k→40k 전체 **+22** (426→448).

| gym | 35k→40k | 40k→45k (사라짐) |
|---|---:|---:|
| **connect4** | **+9** (9→18) | −4 |
| **game2048** | **+9** (17→26) | **−12** |
| gridworld | +4 | −4 |
| cellular_automata | +3 | +1 |
| chess | +3 | −1 |
| debate_judge | +1 | −3 |
| **sokoban** | **−7** | +7 |
| 기타 (nlp/word/ticket/resource) | 0 | 0 |

**상승분 대부분 = connect4 + game2048.**  
45k에서 game2048이 −12로 되돌아가며 spike의 상당 부분이 소멸. sokoban은 반대로 움직이며 aggregate를 가린다.

40k→50k net −14 (주로 game2048 −8, connect4 −6).

---

## 5. Training loss 해석

W&B / 이슈에 정리된 raw 구간 평균 (참고):

| 구간 | 평균 raw loss |
|---:|---:|
| 25–30k | **0.875** |
| 30–35k | 1.149 |
| 35–40k | 1.296 |
| 40–45k | 1.302 |

### 확정: `train/avg_loss`는 누적 평균

코드상 `running / step`. 분모가 커지면 최근 loss가 나빠도 곡선 기울기는 둔해진다.

> `train/avg_loss` 기울기만으로 “회복 중”이라고 해석하면 안 된다.

필요 메트릭: raw / EMA-100 / rolling-500 / 최근 N 평균 / (별도) 누적 평균.

---

## 6. 근본 원인

### 가장 가능성 높은 원인 — **코드로 확정된 메커니즘**

> 첫 학습에서 `--max-steps 20000`으로 scheduler 종료점을 20k에 두어 LR→0이 되게 만든 뒤, resume 시 `--max-steps 60200`으로 horizon을 바꿔 **새** `get_linear_schedule_with_warmup(..., num_training_steps=60200)`을 만들었다.

global step은 ~20k로 이어져 보여도, scheduler의 총 step이 달라져 해당 step의 LR이 ≈`1.48e-5`로 다시 계산된다.

**1차 실수(설계):** 풀 에폭(~60.2k)인데 20k를 schedule 종점으로 잡음.  
**2차 실수(운영):** 끝난 schedule을 같은 W&B run에 horizon만 늘려 “이어 붙임.”

### 확정하지 못하는 세부

당시 프로세스 메모리 안의 tensor를 사후 덤프할 수는 없다. 다만 CLI + 코드 경로 + LR 수치 재현이 일치하므로 원인 메커니즘은 **확정**으로 둔다.

---

## 7. 기여 요인

1. **Physical batch=4, grad accum 없음** — loss variance↑, 높은 LR 재시작 충격↑ (추정 기여)
2. **val n=800** — 54% 부근 단순 SE ≈ ±1.8%p 수준; ±3.5%p는 과대일 수 있으나 표본은 작음
3. **Task 이질성** — connect4/2048↑ vs sokoban↓가 aggregate에 상쇄
4. **`keep_checkpoints=3`** — best@40k 소실 개연성
5. **Provenance 부재** — 로컬 경로만, code/dataset/ckpt artifact 없음, seed=7 단일

---

## 8. 잘된 점

- LR을 history에 남겨 discontinuity 발견 가능
- 5k val로 40k 국소 최고 포착
- task별 val로 spike 분해 가능
- 1k checkpoint로 복구 지점 다수
- 근거 약한 상태에서 52k 중단 → 잔여 ~8k 비용 절약

---

## 9. 실패한 운영 가정

1. **`--resume`이면 같은 학습이 이어진다** → 모델 로딩 ≠ opt/sched/RNG/sampler/LR 연속성  
2. **`train/avg_loss` 평탄화 = 회복** → 누적 평균의 수학적 현상  
3. **단일 val 최고 = 실제 개선** → 40k 56%는 45/50k에서 비재현

---

## 10. 재발 방지 (Adopt)

### P0 — 다음 train 전에

| 조치 | 상태 |
|---|---|
| 풀 horizon을 step 0부터 고정 (`60200` 또는 full epoch) | 정책 |
| resume 시 `max_steps` 불변 + `last_lr` 5% 점프 거절 | 코드 반영 (#8) |
| horizon/peak LR 변경 ⇒ **새 W&B run** | 정책 |
| `best_val` + `pre_resume`를 latest-N 밖에 보존 | 구현 필요 (#9 핵심) |
| raw / EMA / grad_norm 로깅 | 구현 필요 (#10 최소) |

### P1 — 미룸

- 매 ckpt W&B Artifacts 풀 파이프라인 (#11)
- McNemar / per-example dump (#10 후반)
- macro-best 풀 CheckpointManager

### 다음 실험 (#12) — GPU 비었을 때

깨끗한 **20k 가중치** (`systemone-spatial-v2-s1-pre-resume`)를 **init만** 쓰고, **새** opt/sched + **새** W&B run으로 LR pilot A/B/C (`2e-6` / `5e-6` / `1e-5`, bs4×accum4, cosine, 5–10k).  
우승 LR만 3 seed. **`bs62qs8x` 이어 돌리지 말 것.**

---

## 11. 최종 판정

> 모델 발산이나 인프라 고장이 아니다. 그러나 **첫 학습에서 scheduler horizon을 20k로 잘못 잡아 LR→0까지 소진**했고, resume에서 horizon을 60.2k로 바꿔 **LR 시간축이 끊겼다.** 최종 54.25%는 장기 기준선 수준이며, 40k 56%는 connect4·game2048에 치우친 **비지속 spike**다.

### 한 줄 교훈

> **중단 가능한 학습의 핵심은 checkpoint 파일이 아니라, optimizer·scheduler가 같은 시간축 위에 있는지다. Horizon은 step 0에 고정한다.**

### 사용 가능한 산출물

| 산출물 | 사용 |
|---|---|
| `checkpoints/systemone-spatial-v2-s1-pre-resume/` / Hub `dwidlee/systemone-lite-0.5b` | 가중치 init only |
| `step-50k..52k` / post-20k W&B 구간 | 의사결정에 사용 금지 |
| 본 문서 | 재발 방지 + #12 입력 |
