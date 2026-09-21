# 연구 일지 — Phase 2 spatial SFT & JevBench (2026-09-21)

> Lab notebook synthesizing gate postmortem, v2b eval, bare demos, JevBench, and temperature scaling.  
> Claims require the linked JSON reports; demos/GIFs are illustrations only.

| Related | Link |
|---|---|
| Phase 1 note | [`NOTE_MIXED_SFT_2026-09-20.md`](NOTE_MIXED_SFT_2026-09-20.md) |
| v2 postmortem | [`NOTE_SPATIAL_V2_POSTMORTEM.md`](NOTE_SPATIAL_V2_POSTMORTEM.md) |
| Roadmap | [`ROADMAP.md`](ROADMAP.md) · next synth [#7](https://github.com/fritzprix/systemone-lite/issues/7) |
| Gate (v2, old eval) | [`../benchmarks/spatial_v2_report.json`](../benchmarks/spatial_v2_report.json) |
| Gate (v2b, current eval) | [`../benchmarks/spatial_v2b_report.json`](../benchmarks/spatial_v2b_report.json) |
| JevBench | `benchmarks/jevbench_*.json` · temp sweep `jevbench_temp_scale_summary.json` |
| Unofficial rank est. | [`../benchmarks/jevbench_unofficial_rank_estimate.json`](../benchmarks/jevbench_unofficial_rank_estimate.json) |

---

## 0. 한 줄 요약

Phase 2 **alias-CE + bare 2D maps**는 (수정 전 eval에서) 공간 held-out을 크게 올렸다.  
이후 **합성 버그 수정 + 기호 remapping**으로 eval이 바뀌자 같은 가중치의 점수가 급락했다.  
`spatial-v2b`는 v2 continue가 아니라 **base 20k cold-start**라 게이트 실패.  
JevBench에서는 Phase 1 mixed가 정확도·ECE에서 앞서고, **온도 스케일링은 ECE만** 개선한다(순위·정확도는 공식 제출이 아님).

---

## 1. 타임라인 (당일)

| 시각/순서 | 사건 |
|---|---|
| 오전 | `systemone-spatial-v2` 51.2k 완료 (base cold-start, ~204.8k mix) |
| 게이트 | **PASS** on **pre-fix** eval — spatial 0.20→**0.53** vs Phase 1 |
| 오후 | 합성 audit: 고정 스폰, alert 붕괴, `empty_count` 누수 등 → 수정 후 `AUDIT_PASS` |
| 오후 | 기호 remapping (~40%) 도입, train/eval JSONL 재빌드 (HF phase2 업로드는 수정본) |
| 저녁 | `spatial-v2b` 20k 학습 (**`resume=False`**, W&B `s50jkm4d`) |
| 저녁 | v2b 게이트 **FAIL**; bare demos `spatial_v2b/`; v2를 **새 eval**에 재측정 |
| 밤 | JevBench (mixed / v2 / v2b / base); 온도 스윕 T∈{1.0…1.6}; 비공식 순위 추정 |

프로세스 교훈: 긴 GPU 작업은 요청 후에만; mid-ckpt / W&B / sparse val(~5k)은 v2b부터 정착.

---

## 2. 체크포인트 정리

| ID | Init | Steps | 역할 | 게이트 |
|---|---|---:|---|---|
| `systemone-mixed-sft` | base | 10.8k | Phase 1 텍스트+체스 | (Phase 1) |
| **`systemone-spatial-v2`** | base | **51.2k** | Phase 2 **참조** | PASS (옛 eval) |
| `systemone-spatial-v2b` | base | 20k | “continual” 시도였으나 cold-start | **FAIL** (현 eval) |

v2b `train_meta`: gym당 ~10k만 실제 소비(20k×batch4), 풀 204.8k epoch이 아님.

---

## 3. 내부 공간 평가

### 3.1 옛 eval — v2 게이트 (주장 가능했던 숫자)

프로토콜: bare criteria, alias shuffle, no solver override.

| Gym | Phase 1 | **v2** | Δ |
|---|---:|---:|---:|
| Connect4 | 0.158 | **0.754** | +0.60 |
| Sokoban | 0.174 | **0.506** | +0.33 |
| 2048 | 0.188 | **0.502** | +0.31 |
| GridWorld | 0.274 | **0.360** | +0.09 |
| Overall | 0.199 | **0.531** | +0.33 |

Chess shuffled 0.22 (게이트 통과, published 0.24 대비 아슬아슬). General iid/hard **상승**. Invariance OK.

### 3.2 현 eval (수정+remap 후) — 공정 재측정

| Model | Spatial overall | C4 | 2048 | GW | Sokoban |
|---|---:|---:|---:|---:|---:|
| Phase 1 | 0.320 | 0.282 | 0.284 | 0.386 | 0.326 |
| **v2** | **0.346** | 0.408 | 0.260 | 0.396 | 0.320 |
| v2b | 0.329 | 0.406 | 0.226 | 0.374 | 0.308 |

**해석:** eval 분포가 바뀌면 “53% 공간 모델” 서사는 유지되지 않는다.  
v2는 여전히 v2b·P1보다 미세하게 낫지만, **규칙/기호 OOD**에 취약함이 드러남 → Phase 6·#7(다양성) 동기의 실증.

비교글을 v2b에 옛 0.53을 붙인 것은 **오류** (체크포인트 혼동).

### 3.3 Bare demos

경로: `benchmarks/demos/{base_model,sft_model,spatial_v2,spatial_v2b}/`.  
v2b: 2048 score 60 / tile 16; GridWorld early death; Sokoban/C4 미클리어.  
→ held-out top-1 ≠ 짧은 self-play 숙련.

---

## 4. 데이터 / 합성에서 배운 것

| 병리 | 영향 | 조치 |
|---|---|---|
| GridWorld 고정 코너 스폰 | 패턴 암기, GW lift 빈약 | 랜덤 스폰 |
| Sokoban 소형 템플릿 풀 | 중복 trajectory | (부분) 절차 생성; 다양성 여전한 숙제 |
| Alert 클래스 붕괴 / state 누수 | 쉬운 yes/no·정보 누수 | 균형 합성, `empty_count` 제거 |
| 기호 고정 | 글리프 암기 | `symbol_aug` ~40% remap + legend 동기화 |

감사 스크립트: `scripts/audit_phase2_distill.py` — 학습 전 hard-fail 게이트로 유지.

연구 가설과의 연결: **“새 규칙을 읽고 적용”**이 목표면, remapped eval에서의 붕괴는 실패가 아니라 **측정이 맞기 시작한 신호**다.

---

## 5. JevBench (외부 텍스트 의사결정)

동일 조건(경고 I/O 억제): p50 ≈ **13.5–13.7 ms** — 아키텍처 동일하면 지연 차이는 없어야 함.  
과거 16.0 vs 13.7 ms는 **로깅 I/O 착시**.

| Model | Acc (231) | ECE | p50 ms |
|---|---:|---:|---:|
| Qwen base | 39.39% | 0.284 | 13.6 |
| **mixed-sft** | **45.89%** | **0.221** | 13.5 |
| spatial-v2b | 43.72% | 0.306 | 13.7 |
| spatial-v2 | 42.86% | 0.358 | 12.9 |

텍스트 쪽은 Phase 1이 우세. 공간 학습이 JevBench를 자동으로 끌어올리지는 않음(소폭 유지~약하락 + ECE 악화).

### 5.1 온도 스케일링 (T>1 → 과확신 완화)

Acc **불변**(argmax 동일). ECE만 개선.

| T | v2b ECE | v2 ECE |
|---:|---:|---:|
| 1.0 | 0.306 | 0.358 |
| 1.2 | 0.273 | 0.325 |
| 1.4 | 0.243 | 0.301 |
| **1.6** | **0.217** | 0.277 |

mixed ECE(0.221)에 v2b가 닿으려면 주장한 1.2–1.4가 아니라 **≈1.6**.  
“T만으로 리더보드 20위 복귀”는 **미검증**.

### 5.2 리더보드 순위

- 우리 모델은 **공식 JevBench v1.2 표에 없음**.
- 예전의 “mixed = 17위 / 66.8”은 공식 **`kev-0.6b`** 와 혼동.
- 비공식 추정(judge 없음, ECE-only, 로컬 지연, kev급 비용 가정) ≈ mixed **#21 / 64.9**, v2b@T1.6 **#22 / 64.6**. 속도 축이 로컬이라 **낙관적**.

---

## 6. 가설 업데이트

| 가설 | 상태 |
|---|---|
| Bare spatial SFT로 격자 gym top-1 lift | **지지** (옛 eval); 새 분포에선 약화 |
| Phase 1 continue 없이 base+믹스로 general 유지 | **지지** (forget 없음, 오히려 상승 구간 있음) |
| 20k만으로 v2 대체 | **기각** (v2b FAIL) |
| 공간 학습 → JevBench/유창성 일반 향상 | **약하거나 부정** |
| 온도 스케일링 → ECE↓ | **지지**; 순위 주장과 분리할 것 |
| 규칙/기호 다양성이 적응 지능의 축 | **방향 유지** — remap eval 민감도가 근거; #7·Phase 6으로 |

---

## 7. 다음 실험 (우선순위)

1. **`systemone-spatial-v2`를 참조로 유지** — v2b로 교체하지 말 것.  
2. **진짜 continue:** `--resume` from v2 + audited remapped JSONL, mid-eval, 동일 **현 eval**로 재게이트.  
3. **#7 synth diversity:** CA/`n`-horizon + 낱말/언어규칙 게임(~10–20%) — 언어 앵커 + 규칙 다양성.  
4. GridWorld·action-heavy 비중·chess unique 확대.  
5. Phase 3 DPO는 **clean eval 게이트 재통과 후**.  
6. JevBench 공식 순위가 필요하면 프로토콜(judge 티어, 직렬 latency, 비용)대로 제출 — 로컬 추정과 분리.

---

## 8. 아티팩트 체크리스트

- [x] `checkpoints/systemone-spatial-v2` (+ postmortem)  
- [x] `checkpoints/systemone-spatial-v2b` + `spatial_v2b_report.json` (FAIL)  
- [x] demos `spatial_v2` / `spatial_v2b`  
- [x] `audit_phase2_distill.py`, `symbol_aug.py`, rebuilt phase2 JSONL  
- [x] JevBench reports + temp-scale sweep + unofficial rank JSON  
- [ ] v2 resume on remapped data (미실시)  
- [ ] #7 CA / word-game synth (미실시)  
- [ ] spatial-v2 HF 모델 업로드 (TBD)

---

*작성: 2026-09-21. 숫자 갱신 시 이 일지보다 JSON 리포트를 우선한다.*
