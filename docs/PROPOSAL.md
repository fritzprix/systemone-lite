# systemone-lite 기획서 (v3.2)

> **TypeSafe System One HTTP API와 wire-compatible**한 typed decision 서버를  
> causal LM의 **batched next-token + template**으로 구현하는 오픈소스 프로젝트.  
> 아키텍처·가중치·RLCD는 복제하지 않음. **요청/응답 JSON 계약만 호환.**

| 항목 | 내용 |
|---|---|
| **프로젝트명** | systemone-lite |
| **목표** | `POST /v1/systemone` 가 Jev와 동일한 request/response shape |
| **호환 기준** | [TypeSafe API reference](https://docs.typesafe.ai/api.md) (공개 스펙) |
| **핵심 수단** | Causal LM + prompt template + option-constrained logits + batching |
| **백본 정책** | **최대한 경량 우선** — 품질은 나중에 키움 |
| **기본 백본** | `Qwen/Qwen2.5-0.5B-Instruct` (Apache 2.0) |
| **비목표** | Jev 샘플러/가중치 복제, 챗 생성, BERT MLM |

---

## 1. 문제 정의

에이전트·워크플로는 **문자열 파싱 없이** 분기할 수 있는 typed decision이 필요하다.  
TypeSafe Jev는 이를 `POST /v1/systemone` 계약으로 공개했다.

본 프로젝트 목표:

1. **같은 curl body**로 우리 서버를 호출할 수 있을 것  
2. **같은 `answers` shape**로 기존 Jev 연동 코드가 (base URL만 바꿔) 동작할 것  
3. 내부 구현은 오픈 causal LM + batch AR

```text
호환:   HTTP path, request JSON, response JSON, 질문 3종(noul/choice/score)
비호환: 모델 품질, latency, auth 벤더, 내부 샘플러, RLCD
```

---

## 2. 핵심 설계 (확정)

### 2.1 추론 패턴: Batched AR, not MLM

**하지 않는 것**

- 한 시퀀스에 `[MASK]` 여러 개 → MLM식 추출  
- GPT-2 logits를 “해당 position 토큰”으로 해석 (GPT-2는 **다음 토큰** 예측)

**하는 것**

```text
B1:  [state][template(q1)]  →  option-constrained next-token probs
B2:  [state][template(q2)]  →  option-constrained next-token probs
B3:  [state][template(q3)]  →  option-constrained next-token probs
        └── GPU batch (이후 prefix KV cache)
```

- 질문마다 독립 causal forward → Jev의 “질문 독립 평가” **계약**과 정합  
- 병렬성은 배치/캐시에서 나옴 (전용 parallel sampler 주장 안 함)

### 2.2 질문 타입 → 내부 매핑

| API type | 내부 처리 | Answer 조립 |
|---|---|---|
| **noul** | 2-way `{true, false}` (또는 criteria 라벨) constrained softmax | `noul = P(true)` |
| **choice** | `criteria` 키를 옵션으로 constrained softmax | `choice`, `probabilities`, `confidence` |
| **score** | levels `0..L-1` constrained softmax | `score=Σ i·p_i`, `legend`, `probabilities`, `confidence` |

### 2.3 Choice / Score 확률

옵션(또는 level 라벨)이 **단일 토큰**이거나 **단일 토큰 alias**로 매핑될 때:

```python
logits = model(input_ids).logits[:, -1, :]
option_logits = logits[:, option_token_ids]
probs = softmax(option_logits, dim=-1)
```

다중 토큰 키(`escalate` 등):

- Phase 1: criteria 키를 프롬프트에 쓰고, **답은 인덱스/단축 토큰**(`0`,`1`,`billing`이 한 토큰이면 그대로)  
- Phase 2: 옵션별 sequence logprob 합산

### 2.4 Confidence (Choice / Score만)

공식 문서는 “분포 shape에서 유도”라고만 하고 수식은 비공개에 가깝다.  
공개 설명·서드파티 정리와 일치하는 **기본 구현** (검증 후 고정):

```text
n = number of options (or levels)
p_max = max(probabilities)
confidence = (p_max - 1/n) / (1 - 1/n)     # n>=2, clip to [0,1]
```

예: n=3, p_max=0.8 → (0.8−⅓)/(⅔) ≈ 0.70  

- Noul 답변에는 `confidence` **없음** (공식 스펙과 동일)  
- 수식이 TypeSafe와 1:1이 아님을 README에 명시. 동일 공식이 확인되면 맞춘다.  
- 클라이언트는 항상 `probabilities`로 자체 통계를 계산할 수 있음

### 2.5 Score 값

```text
legend: { "0": criteria[0], "1": criteria[1], ... }
probabilities: { "0": p0, "1": p1, ... }   # keys are strings, sum≈1
score: Σ_i (i * p_i)                        # float, levels 사이에 올 수 있음
```

### 2.6 Prompt template (초안)

```text
### State
{state_rendered}

### Question
{instructions}

### Criteria
{criteria_rendered}

### Answer
```

- `state`: string이면 그대로, object/array면 `json.dumps(..., ensure_ascii=False, indent=2)`  
- Choice criteria: `option: description` 목록 (`null` description 허용)  
- Score criteria: 순서 있는 level 목록  
- Noul criteria: optional `true`/`false` 설명  
- **question map의 key는 모델에 넣지 않음** (공식 스펙: key는 id일 뿐 inference에 미사용)

### 2.7 Prompt template — Answer 슬롯

모델이 뱉을 심볼과 API 키를 분리:

| type | 모델이 예측하는 심볼 | API로 노출 |
|---|---|---|
| noul | `yes` / `no` (단일 토큰) | `noul=P(yes)` |
| choice | criteria **key** (가능하면 단일 토큰) 또는 `A`/`B`/… 매핑 | key → `choice` / `probabilities` |
| score | level index 토큰 `0`/`1`/… | weighted `score` + legend |

### 2.8 백본: 경량 우선

```text
원칙: 가장 작은 Instruct 모델로 파이프라인·API·학습 루프를 먼저 완성한다.
      정확도가 막히면 같은 코드로 모델 id만 키운다.
```

| 역할 | 모델 | 비고 |
|---|---|---|
| **default** | `Qwen/Qwen2.5-0.5B-Instruct` | ~0.5B, Apache 2.0, CPU/소형 GPU 가능 |
| smoke 대안 | `Qwen/Qwen3-0.6B` (thinking **off**) | 동급 경량; Qwen3면 non-thinking 필수 |
| scale-up 1 | `Qwen/Qwen2.5-1.5B-Instruct` | default가 품질 천장일 때 |
| scale-up 2 | `Qwen/Qwen3-4B-Instruct-2507` | 벤치/데모용 |
| legacy | `gpt2` / `gpt2-medium` | 회귀·비교용, 기본값 아님 |

경량 선택 이유:

1. System One은 **긴 생성**이 아니라 1-step decision → 초대형 모델 이득이 작음  
2. 배치 질문 수가 늘수록 **작은 모델 + prefix cache**가 latency에 유리  
3. FT·실험 반복 비용 최소  
4. API/스키마 작업과 모델 품질 작업을 분리 가능  

VRAM 대략 (fp16, infer only): 0.5B ≈ **1–2GB** 급.  
양자화(int4/int8)는 Phase 2+ 옵션.

---

## 3. API 호환 계약 (Source of Truth)

기준 문서: https://docs.typesafe.ai/api.md  

### 3.1 Endpoint

```http
POST /v1/systemone
Authorization: Bearer <API_KEY>    # 로컬/셀프호스트: 설정 가능, PoC는 생략 옵션
Content-Type: application/json
```

호환을 위해 **path와 method를 고정**. 추가 엔드포인트(` /health`)는 허용하되 System One 계약 밖.

### 3.2 Request body

```json
{
  "model": "systemone-lite-latest",
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this convey urgency?",
      "criteria": {
        "true": "Explicitly time-sensitive",
        "false": "No urgency expressed"
      }
    },
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts"
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}
```

| 필드 | 타입 | 규칙 |
|---|---|---|
| `model` | string | alias: `systemone-lite-latest` → 기본 백본(`Qwen2.5-0.5B-Instruct`). concrete HF id도 허용. `jev-*`는 remap 또는 422(문서화) |
| `state` | string \| object \| array | 텍스트 또는 JSON 직렬화 |
| `questions` | **map** (array 아님) | 키 = 답변 id. **422** if array |
| `questions.*.type` | `noul` \| `choice` \| `score` | |
| `questions.*.instructions` | string | 필수 |
| `questions.*.criteria` | type별 | noul: optional `{true?, false?}`; choice: **required** map (값 string\|null); score: **required** ordered array, length 2–10 |

**검증 실패 → HTTP 422** (공식과 동일 계열).

### 3.3 Response body

```json
{
  "model": "systemone-lite-0.1.0",
  "answers": {
    "is_urgent": {
      "type": "noul",
      "noul": 0.92
    },
    "department": {
      "type": "choice",
      "choice": "technical",
      "probabilities": {
        "billing": 0.08,
        "technical": 0.85,
        "sales": 0.07
      },
      "confidence": 0.775
    },
    "frustration": {
      "type": "score",
      "score": 1.6,
      "legend": {
        "0": "Calm",
        "1": "Frustrated",
        "2": "Very angry"
      },
      "probabilities": {
        "0": 0.05,
        "1": 0.3,
        "2": 0.65
      },
      "confidence": 0.475
    }
  },
  "usage": {
    "input_tokens": 312,
    "output_tokens": 0
  }
}
```

| 규칙 | 내용 |
|---|---|
| `answers` 키 | request `questions` 키와 **동일** |
| 각 answer `type` | 질문 type과 동일 |
| Noul | `{ type, noul }` only — **confidence 없음** |
| Choice | `{ type, choice, probabilities, confidence }` — probs 키 = criteria 키, sum≈1 |
| Score | `{ type, score, legend, probabilities, confidence }` — probs/legend 키 = `"0".."L-1"` |
| `usage.output_tokens` | decision API이므로 **0**이 자연스러움 (Jev도 예시에서 0) |
| `model` | **실제 응답한 concrete id** (alias가 아닌 resolved id) |

### 3.4 에러

| Status | 의미 |
|---|---|
| 401 | API key 설정이 켜져 있는데 없거나 잘못됨 |
| 422 | body 검증 실패 (choice criteria 누락, score &lt; 2 levels, questions가 array 등) |
| 429 | (선택) rate limit |
| 500 | 내부 오류 |

에러 body는 최소한 `{ "detail": ... }` 또는 TypeSafe류 메시지. **완벽 동일은 비목표**, status code 의미는 맞춤.

### 3.5 호환성 레벨

| Level | 의미 | 목표 phase |
|---|---|---|
| **L0** | OpenAPI/JSON schema가 공식 예시와 일치, 픽스처 round-trip | Phase 1 |
| **L1** | curl로 공식 예시 body → 우리 서버 → shape-valid response | Phase 1 |
| **L2** | 기존 앱이 `base_url`만 바꿔 동작 (auth 설정 시) | Phase 2 |
| **L3** | 공식 `typesafe_sdk` / `@typesafe-ai/sdk`를 base URL 오버라이드로 사용 | 도전 과제 (SDK가 URL 고정이면 어댑터) |

**호환성 테스트**: 공식 문서 예시를 fixtures로 넣고 request validate + response jsonschema assert.

### 3.6 의도적으로 다른 점 (문서화 필수)

| 항목 | TypeSafe Jev | systemone-lite |
|---|---|---|
| 모델 id | `jev-latest` 등 | `systemone-lite-*` (내부는 Qwen 0.5B 등) |
| 품질·속도·가격 | 벤더 주장 | **경량 우선** — 자체 벤치 |
| confidence 수식 | 미완전 공개 | §2.4 공식 (명시) |
| auth | `TYPESAFE_API_KEY` | 자체 키 또는 없음 |
| 최대 choice 카드 | 255 (고카드 2-stage) | PoC는 작은 N; 이후 확장 |

---

## 4. 시스템 아키텍처

```text
POST /v1/systemone
        │
        ▼
┌─ Schema (Pydantic) ─────────────────────────────────────┐
│  SystemOneRequest → validate questions map & criteria   │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─ PromptBuilder ─────────────────────────────────────────┐
│  state render + per-question template → input_ids       │
│  option/level → token id table                          │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─ BatchedInferencer ─────────────────────────────────────┐
│  forward last-token logits (batch over questions)       │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─ AnswerAssembler ───────────────────────────────────────┐
│  noul / choice / score + confidence + usage             │
└───────────────────────────┬─────────────────────────────┘
                            ▼
              SystemOneResponse JSON
```

### 4.1 모듈

| 모듈 | 책임 |
|---|---|
| `schema` | Request/Response/Question/Answer — **Jev-shaped** Pydantic |
| `prompt` | state render, criteria render, token 매핑 |
| `infer` | batched forward, token counts |
| `assemble` | probs → noul/choice/score/confidence |
| `api` | FastAPI `POST /v1/systemone` |
| `client` | 얇은 Python client (`system_one(...)`) — SDK 흉내 |
| `train` | CE/Brier, ECE |
| `compat` | fixtures + jsonschema tests vs 공식 예시 |

### 4.2 레포 구조

```text
systemone-lite/
├── README.md
├── LICENSE
├── pyproject.toml
├── docs/
│   └── PROPOSAL.md
├── openapi/
│   └── systemone.yaml          # Jev-compatible OpenAPI
├── configs/
│   ├── infer.yaml
│   └── train.yaml
├── src/systemone_lite/
│   ├── __init__.py
│   ├── schema.py
│   ├── prompt.py
│   ├── infer.py
│   ├── assemble.py
│   ├── confidence.py
│   ├── api.py
│   ├── client.py
│   └── train/
├── tests/
│   ├── fixtures/               # 공식 문서 기반 request/response
│   ├── test_schema.py
│   ├── test_compat_shape.py
│   └── test_confidence.py
└── scripts/
    ├── demo_systemone.py
    └── train.py
```

---

## 5. 클라이언트 스케치

```python
from systemone_lite import SystemOneClient

client = SystemOneClient(base_url="http://localhost:8000", api_key=None)

response = client.system_one(
    model="systemone-lite-latest",
    state="My card was charged twice...",
    questions={
        "needs_review": {
            "type": "noul",
            "instructions": "Does this ticket need a human agent?",
            "criteria": {
                "true": "money, legal, or unanswered complaint",
                "false": "routine question a bot can close",
            },
        },
        "route": {
            "type": "choice",
            "instructions": "Route this ticket to a team.",
            "criteria": {
                "billing": "payment or charge problems",
                "shipping": "delivery problems",
                "technical": "application bugs",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": ["low", "medium", "high"],
        },
    },
)

assert response.answers["needs_review"].noul >= 0.0
assert response.answers["route"].choice in {"billing", "shipping", "technical"}
assert 0.0 <= response.answers["urgency"].score <= 2.0
```

헬퍼 (SDK 감각):

```python
from systemone_lite import noul, choice, score

questions = {
    "needs_review": noul("Does this need a human?", criteria={...}),
    "route": choice("Route to team", criteria={...}),
    "urgency": score("Urgency", criteria=["low", "medium", "high"]),
}
```

---

## 6. 학습 전략

1. Supervised CE on option/level tokens (noul=binary)  
2. ECE / Brier / reliability diagram 상시 측정  
3. “RLCD 재현” 표기 금지  
4. 합성 데이터는 **공식 question shape**으로 생성 (`type`/`instructions`/`criteria`)

---

## 7. Phase별 범위 (API 호환 우선)

### Phase 0 — 기획 (현재)

- [x] MLM 폐기, batched AR 확정  
- [x] **Jev wire-compat**을 1급 요구사항으로 고정  
- [ ] fixtures + OpenAPI 초안 합의  

### Phase 1 — Compat PoC (1–2주)

**백본: `Qwen2.5-0.5B-Instruct`만** (다른 모델은 config 슬롯만 열어 둠)

**반드시**

- Pydantic schema = §3  
- `POST /v1/systemone`  
- noul + choice + score **모두 응답 shape 충족** (품질은 낮아도 됨)  
- confidence §2.4  
- `tests/test_compat_shape.py` (공식 예시 body)  
- single-token / index-token 경로  
- `usage.input_tokens` 계산  
- 로컬에서 **작은 VRAM/가능하면 CPU**로 demo 가능  

**나중**

- 1.5B+ 업스케일, prefix cache, 대규모 FT  

### Phase 2 — Quality + L2

- 합성 데이터 FT, ECE 개선  
- multi-token option 전략  
- base_url 스왑 연동 가이드  
- latency 최적화  

### Phase 3 — Serve & OSS

- 문서, MIT, 예제, (선택) 가중치 공개  

### Phase 1 성공 기준

| 메트릭 | 목표 |
|---|---|
| Fixture request → 200 + schema-valid response | 100% |
| Schema violation (criteria 밖 label) | 0% (구조적으로 불가) |
| Noul에 confidence 필드 없음 | assert |
| Score `legend`/`probabilities` 키 `"0".."L-1"` | assert |
| 공식 예시와 다른 커스텀 필드 | response에 **추가하지 않음** (strict) |

---

## 8. 리스크

| 리스크 | 대응 |
|---|---|
| confidence 수식 불일치 | probs는 호환; confidence는 문서화 + 추후 정렬 |
| criteria 키가 multi-token | alias 테이블 / index answers |
| choice 카드 255 | PoC 상한(예: 32) + 422; 이후 2-stage |
| SDK URL 고정 | 자체 client + HTTP 호환으로 L2 충족 |
| `jev-*` model 문자열 | alias remap 또는 422 + 명확 메시지 |

---

## 9. 예산·일정

이전 v2와 동일 대략치 ($350–1,500, 4–8주).  
Phase 1 무게중심이 “모델 성능” → **“스키마 호환 + 엔드포인트”** 로 이동해 PoC 기간 단축 가능.

---

## 10. 결정 로그

| 날짜 | 결정 |
|---|---|
| 2026-09-19 | MLM / multi-mask **폐기** |
| 2026-09-19 | 추론 = batched causal next-token + template |
| 2026-09-19 | **API = TypeSafe `POST /v1/systemone` wire-compatible** |
| 2026-09-19 | 질문 3종 `noul` / `choice` / `score` 전부 Phase 1 shape 지원 |
| 2026-09-19 | questions는 **map**; array면 422 |
| 2026-09-19 | confidence = `(p_max - 1/n)/(1 - 1/n)` (명시적; 벤더 수식과 다를 수 있음) |
| 2026-09-19 | response에 커스텀 필드 추가 금지 (strict compat) |
| 2026-09-19 | “RLCD 재현” 표현 금지 |
| 2026-09-19 | 백본 **경량 우선**: default = `Qwen/Qwen2.5-0.5B-Instruct` |
| 2026-09-19 | GPT-2는 legacy 비교용; 기본값 아님 |
| 2026-09-19 | 프로젝트명 **gpt2-jev → systemone-lite** (패키지 `systemone_lite`) |

---

## 11. 다음 액션

1. 본 v3(경량 백본) 합의  
2. `openapi/systemone.yaml` + `schema.py` + fixtures  
3. FastAPI stub (검증만, 더미 probs) → shape 테스트 그린  
4. `Qwen2.5-0.5B-Instruct` batched infer 연결  
5. `scripts/demo_systemone.py` (공식 예시 body)

---

*문서 버전: v3.2 — 2026-09-19*  
*v3.1 대비: 프로젝트명 **systemone-lite** 로 변경.*  
*참고: https://docs.typesafe.ai/api.md*
