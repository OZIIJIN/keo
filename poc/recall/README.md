## 목표

`poc/recall`은 KEO recall retrieval과 memory graph 구조를 검증하는 실험 디렉토리다.

현재 이 PoC에서 보는 질문은 크게 두 가지다.

1. **Retrieval baseline**
   - 한국어 메모에서 text dense retrieval이 기본적으로 어느 정도 동작하는가
   - relation tag가 retrieval signal로 실제 도움이 되는가
   - text channel과 relation channel을 분리했을 때 성능이 유지되거나 좋아지는가
   - query expansion이나 LLM reranker가 기대한 만큼 retrieval 성능을 끌어올리는가

2. **Memory graph**
   - KEO다운 연결이 단순 문장 유사도가 아니라 반복 패턴, 반응 방식, 회고 역할 연결이라면 어떤 데이터 구조가 필요한가
   - relation tag를 node로 만들지 않고 routing metadata로만 써도 semantic memory node를 잘 만들 수 있는가
   - LLM을 top-k reranker가 아니라 memory maintainer로 썼을 때 `memory_node`와 `memory_evidence`가 쓸 만하게 축적되는가

핵심 결론은 `relation_tag`는 최종 node가 아니라 **메모에 붙는 고정 라벨이자 내부 routing signal**이고, 실제 축적 단위는 **semantic memory node**라는 점이다.

---

## 디렉토리 구조

현재 기준 구조는 아래처럼 둔다.

```text
poc/recall/
├── main.py
├── memory_graph.py
├── data/
│   ├── memos.json
│   ├── memo_annotations.json
│   ├── memory_nodes.json
│   ├── memory_evidence.json
│   └── memory_links.json
├── data_eval/                  # build-eval-graph 실행 결과
│   ├── memos.json
│   ├── memo_annotations.json
│   ├── memory_nodes.json
│   ├── memory_evidence.json
│   └── memory_links.json
└── eval/
    ├── past_memos.json
    ├── past_memo_tags.json
    ├── eval_cases.json
    └── eval_case_tags.json
```

역할은 아래와 같다.

| 경로 | 역할 |
| --- | --- |
| `main.py` | dense / relation / RRF / reranker retrieval eval runner |
| `memory_graph.py` | semantic memory graph 생성, 저장, 평가 |
| `eval/past_memos.json` | 500개 과거 메모 |
| `eval/past_memo_tags.json` | 과거 메모별 fixed relation tags |
| `eval/eval_cases.json` | 평가 케이스 |
| `eval/eval_case_tags.json` | 평가 케이스별 fixed relation tags |
| `data/` | 실제 ingest 실험용 기본 상태 저장소 |
| `data_eval/` | eval dataset으로 생성한 semantic graph 상태 저장소 |

`eval/`은 입력 데이터이고, `data_eval/`은 생성된 memory graph 결과물이다.  
두 디렉토리를 분리해야 기존 데이터와 eval 실험 상태가 섞이지 않는다.

---

## 데이터셋

현재 데이터셋은 KEO 스타일의 메모를 다양하게 포함한다.

- 짧은 자기관찰 메모
- 생활 행정 일 미룸 메모
- 일정 전후 한 줄 메모
- 책/문장/인용이 오래 남는 메모
- 제품 아이디어와 일상 메모가 같은 흐름에 섞이는 메모
- 에너지 저하, 리듬 붕괴, 답장 미룸, 예약 미룸 같은 생활성 메모
- 장보기, 빨래, 분리수거, 충전기, 우산, 약, 물 마시기 같은 일상 관리 메모
- 공간, 이동, 날씨, 소비, 관계 온도, 읽기 기록, 제품 관찰 같은 비회피성 일상 메모

의도적으로 너무 템플릿처럼 보이지 않게, 실제 사람이 그때그때 남겼을 법한 질감으로 섞어둔다.

---

## Relation Retrieval용 Eval 원칙

이 eval은 완전 동일문장 검색이 아니다.

목적은 아래 연결을 복원하는 retrieval이 가능한지 보는 것이다.

- 같은 미룸 패턴
- 같은 일정 여파
- 같은 문장 여운
- 같은 자기속임 구조
- 같은 생활성 unfinished loop
- 같은 제품적 자기이해 관찰

### 1. 같은 표현보다 같은 패턴 역할을 우선한다

정답은 "같은 단어를 쓴 메모"가 아니라 아래 중 하나면 포함 가능하다.

- 같은 반응 방식
- 같은 판단 흐름
- 같은 회고 역할
- 같은 여파를 설명하는 메모

### 2. 같은 memo type이 아니어도 정답이 될 수 있다

KEO에서는 아래처럼 서로 다른 표면 타입도 같은 역할이면 연결 대상이 된다.

- quote 메모 ↔ 그 문장이 하루 판단에 남은 메모
- 생활 일 미룸 메모 ↔ 그 미룸이 자존감에 남는 메모
- 제품 아이디어 메모 ↔ 자기이해 구조를 제품으로 옮기려는 메모

즉 `memo type` 일치보다 `회고/패턴 역할` 일치를 더 우선한다.

### 3. expected_ids는 대표 예시만 두지 않는다

기존 miss의 큰 원인 중 하나는 정답셋이 특정 하위 표현군에만 묶여 있던 점이었다.

그래서 relation eval에서는 다음을 반영한다.

- 같은 cluster 안의 다른 자연스러운 변형 메모도 포함
- 너무 특정 표현에만 편향된 정답셋은 확장
- 한 패턴을 설명하는 생활형/회고형/인용형 메모를 함께 허용

---

## 현재 retrieval baseline 구조

`main.py`는 retrieval eval runner로 유지한다.

현재 지원하는 retrieval mode는 아래와 같다.

- `text`
  - 메모 원문만 embed해서 retrieval
- `relation`
  - relation tag만 embed해서 retrieval
- `rrf`
  - text ranking과 relation ranking을 따로 만든 뒤 RRF로 fuse
- `relation_gate`
  - relation retrieval로 후보군을 먼저 줄인 뒤 text dense로 rerank
- `text_rerank`
  - text dense 후보군을 먼저 뽑은 뒤 LLM으로 최종 순서를 다시 정렬

지금은 `원문 + 태그를 한 문자열로 합쳐 한 번에 embed`하는 방식보다, 채널을 분리해서 다루는 쪽을 우선 본다.

---

## 현재까지의 retrieval 결론

현재까지 확인한 방향은 아래와 같다.

- dense-only baseline은 여전히 필요하다
- relation tag는 retrieval signal로 실제 도움이 된다
- query expansion / LLM reranker는 기대한 효과보다 현저히 낮았다
- best case도 hit rate가 약 49% 수준이라 retrieval ranking만 더 복잡하게 만드는 것으로는 한계가 있다
- LLM reranker는 사람이 보기엔 그럴듯한 패턴 후보를 올리지만, eval expected hit 기준에서는 좋은 dense 후보를 밀어내는 경우가 있었다
- LLM은 top-k reranker보다 memory maintainer로 쓰는 쪽이 KEO 제품 방향에 더 맞다
- eval miss 중 일부는 retrieval 실패보다 `expected_ids` undercoverage 문제였다

핵심은 KEO retrieval이 단순 문장 유사도보다 `반복 패턴`, `반응 방식`, `회고 역할` 연결에 더 가깝다는 점이다.

---

## 현재 결과 스냅샷

기준:

- dataset: `past_memos.json` 500개, `eval_cases.json` 142개
- top-k: 10
- embedding model: `qwen3-embedding`
- eval expected ids: 707개

현재 dense baseline 결과와 일상 메모 추가 전 reranker audit 결과:

| mode | hits | expected | hit rate | zero-hit |
| --- | ---: | ---: | ---: | ---: |
| `text` | 247 | 707 | 34.94% | 28 |
| `text_rerank` + `qwen3:8b` (`candidate_k=30`, 일상 메모 추가 전) | 208 | 527 | 39.47% | 13 |

이 결과는 memory graph 도입 전 baseline 해석용이다.

---

# Memory Graph PoC

`memory_graph.py`는 KEO의 memory graph 구조를 검증하는 실험이다.

이제 memory graph는 seed node를 만들지 않는다.

## 최종 모델

```text
memo
= 사용자가 쓴 원본 메모

memo_annotations
= 메모별 1차 해석
= relation_tags 저장
= routing metadata 역할

memory_nodes
= semantic node만 저장
= 여러 메모가 쌓여 드러난 사용자 이해 카드

memory_evidence
= memory_node와 memo의 근거 연결

memory_links
= semantic memory_node 간 관계
```

### relation_tag는 node가 아니다

`relation_tag`는 메모 하나에 붙는 고정 라벨이다.

예:

```json
{
  "memo_id": "memo-006",
  "relation_tags": ["start_avoidance", "overplanning", "avoidance_rationalization"]
}
```

이 tag를 `node-start-avoidance` 같은 node로 만들지 않는다.

대신 relation tag는 아래 용도로만 쓴다.

- 기존 semantic node 후보를 좁히기
- dense로 찾은 과거 memo와 함께 LLM judge context 구성
- 새 node를 만들 때 relation_tags 필드로 남기기
- eval에서 query와 node/evidence 간 routing signal로 사용하기

즉 relation tag는 **routing metadata**이고, 최종 축적 단위는 semantic memory node다.

### semantic memory node

semantic node는 여러 메모를 근거로 만들어진 자기이해 카드다.

예:

```json
{
  "id": "node-pattern-abc12345",
  "source": "llm",
  "type": "pattern",
  "title": "시작을 준비 작업으로 우회함",
  "summary": "중요한 일을 바로 시작하지 않고 자료 조사나 구조 정리로 미루는 흐름이 반복된다.",
  "relation_tags": ["start_avoidance", "overplanning"],
  "representative_evidence_ids": ["memo-006", "memo-022"],
  "evidence_count": 2,
  "confidence": 0.6,
  "last_seen_at": "2026-06-02T00:00:00Z",
  "created_at": "2026-06-02T00:00:00Z",
  "updated_at": "2026-06-02T00:00:00Z"
}
```

`source`는 이 node가 어떻게 생성됐는지 나타내는 출처다.  
semantic 여부는 `source == "llm"`으로 판단하지 않고, `type/title/summary/relation_tags` 구조가 유효한지로 판단한다.

현재 semantic node type은 아래 네 가지다.

```python
SEMANTIC_NODE_TYPES = {"pattern", "state", "question", "product_insight"}
```

---

## Memory Graph 처리 흐름

새 메모 저장 시 흐름은 아래와 같다.

```text
새 메모 저장
-> relation tag 분류
-> memo_annotations에 relation_tags 저장
-> dense retrieval로 관련 과거 메모 검색
-> relation tag overlap + dense evidence hit로 semantic memory_node 후보 계산
-> LLM이 attach/create/attach_and_create/ignore 판단
-> 새 메모와 근거 메모를 memory_evidence로 연결
-> node evidence_count, confidence, last_seen_at 갱신
```

각 요소의 역할은 아래처럼 본다.

| 요소 | 역할 |
| --- | --- |
| dense retrieval | 관련 과거 메모를 찾는 evidence finder |
| relation tag | memory node 후보를 좁히는 routing signal |
| LLM | top-k reranker가 아니라 node 연결 판단과 node 생성 판단을 맡는 memory maintainer |
| memory_node | 여러 메모가 쌓여 드러난 반복 패턴, 상태, 질문, 제품 통찰 카드 |
| memory_evidence | memory_node를 뒷받침하는 실제 메모 연결 |
| memory_links | 반복적으로 함께 나타난 memory_node 간 관계 |

---

## Memory Graph MVP가 하는 것

현재 MVP는 아래까지만 한다.

- 새 메모를 `data/memos.json`에 저장
- relation tag를 `data/memo_annotations.json`에 저장
- dense retrieval로 비슷한 과거 메모를 찾기
- 이미 쌓인 semantic node를 후보로 계산
- LLM이 `attach / create / attach_and_create / ignore` 판단
- 선택된 semantic node의 `memory_evidence` 갱신
- node `evidence_count`, `confidence`, `last_seen_at` 갱신
- 충분히 쌓인 evidence를 바탕으로 `memory_links` 후보 생성

아직 하지 않는 것:

- LLM이 node merge/split을 자동으로 확정
- LLM이 모든 node link relation을 최종 판정
- 앱 backend API 연결
- DB 트랜잭션/동시성 제어

---

## Memory Graph 저장 파일

기본 ingest 결과는 `data/`에 저장된다.

```text
data/
├── memos.json
├── memo_annotations.json
├── memory_nodes.json
├── memory_evidence.json
└── memory_links.json
```

eval dataset 기반 graph 생성 결과는 `data_eval/`에 저장하는 것을 권장한다.

```text
data_eval/
├── memos.json
├── memo_annotations.json
├── memory_nodes.json
├── memory_evidence.json
└── memory_links.json
```

---

# 실행 방법

## 전제

Ollama가 실행 중이어야 한다.

```bash
ollama serve
```

필요한 모델이 로컬에 있어야 한다.

```bash
ollama pull qwen3-embedding
ollama pull qwen3:8b
```

프로젝트 디렉토리로 이동한다.

```bash
cd poc/recall
```

---

## 1. Retrieval baseline 실행

### text-only 검색

```bash
python3 main.py \
  --memo "회의와 알림에 계속 끌려다녀서 중요한 작업을 제대로 못 했다." \
  --model qwen3-embedding \
  --retrieval-mode text \
  --top-k 10
```

### relation-only 검색

```bash
python3 main.py \
  --memo "완벽하게 하려다가 또 시작을 미루는 중이다. 일단 거칠게라도 초안부터 만들어야 할 듯." \
  --model qwen3-embedding \
  --retrieval-mode relation \
  --relation-tags start_avoidance,evaluation_fear
```

### RRF 검색

```bash
python3 main.py \
  --memo "완벽하게 하려다가 또 시작을 미루는 중이다. 일단 거칠게라도 초안부터 만들어야 할 듯." \
  --model qwen3-embedding \
  --retrieval-mode rrf \
  --relation-tags start_avoidance,evaluation_fear
```

### LLM으로 relation tag를 뽑아 검색

```bash
python3 main.py \
  --memo "완벽하게 하려다가 또 시작을 미루는 중이다. 일단 거칠게라도 초안부터 만들어야 할 듯." \
  --model qwen3-embedding \
  --retrieval-mode rrf \
  --tagging-mode llm
```

### text eval

```bash
python3 main.py \
  --eval \
  --model qwen3-embedding \
  --retrieval-mode text \
  --top-k 10 \
  --summary-only
```

### relation eval

```bash
python3 main.py \
  --eval \
  --model qwen3-embedding \
  --retrieval-mode relation \
  --tagging-mode fixed \
  --top-k 10 \
  --summary-only
```

### RRF eval

```bash
python3 main.py \
  --eval \
  --model qwen3-embedding \
  --retrieval-mode rrf \
  --tagging-mode fixed \
  --top-k 10 \
  --summary-only
```

### relation gate eval

```bash
python3 main.py \
  --eval \
  --model qwen3-embedding \
  --retrieval-mode relation_gate \
  --relation-candidate-k 30 \
  --tagging-mode fixed \
  --top-k 10 \
  --summary-only
```

### LLM reranker eval

```bash
python3 main.py \
  --eval \
  --model qwen3-embedding \
  --retrieval-mode text_rerank \
  --rerank-model qwen3:8b \
  --rerank-candidate-k 30 \
  --tagging-mode fixed \
  --top-k 10 \
  --summary-only
```

---

# Memory Graph 실행 방법

## 1. 단건 ingest

fixed relation tags를 직접 넣어 ingest한다.

```bash
python3 memory_graph.py ingest \
  --memo "초안 쓰려다가 또 자료만 찾고 있다." \
  --relation-tags start_avoidance,overplanning \
  --use-llm-node-judge \
  --dense-top-k 5
```

LLM으로 relation tag를 분류해서 ingest한다.

```bash
python3 memory_graph.py ingest \
  --memo "답장 하나 못 보냈는데 하루 종일 마음에 걸렸다." \
  --tagging-mode llm \
  --tagging-model qwen3:8b \
  --use-llm-node-judge \
  --dense-top-k 5
```

주의: `--use-llm-node-judge`를 빼면 semantic node attach/create 판단을 하지 않는다.  
현재 seed node가 없기 때문에 이 옵션을 빼면 `memory_nodes`가 거의 생기지 않을 수 있다.

---

## 2. 현재 memory node 확인

```bash
python3 memory_graph.py nodes --limit 10
```

---

## 3. memory link 후보 생성

기본 `build-links`는 같은 메모에 반복적으로 같이 붙은 node pair를 `overlaps`로 저장한다.

```bash
python3 memory_graph.py build-links \
  --min-support 3 \
  --limit 20
```

시간 순서 기반 `follows` 후보는 noise가 많을 수 있어서 명시적으로 켤 때만 사용한다.

```bash
python3 memory_graph.py build-links \
  --include-temporal \
  --min-support 3 \
  --limit 20
```

---

## 4. memory link 확인

```bash
python3 memory_graph.py links --limit 10
```

---

## 5. 기본 data 상태 audit

현재 `audit`은 기본 `data/` 상태를 본다.

```bash
python3 memory_graph.py audit
```

현재 `audit`에서 자동 측정하는 지표는 아래다.

- `attachment_coverage`: 전체 메모 중 하나 이상의 semantic node에 붙은 메모 비율
- `attached_pair_count`: `(memo_id, node_id)` evidence pair 수
- `attached_memo_count`: semantic node에 붙은 memo 수
- `source_counts`: evidence source 분포
- `avg_nodes_per_memo`: 메모 하나가 평균 몇 개 node에 붙는지
- `dense_added_node_ratio`: 전체 evidence 중 dense 결과로 추가 연결된 비율
- `node_coherence`: node tag와 evidence memo tag가 겹치는 비율
- `evidence_count_distribution`: node별 evidence 쏠림 정도
- `orphan_node_count`: evidence가 적은 node 수
- `over_broad_node_count`: evidence가 과도하게 몰린 node 수
- `support_count_distribution`: link가 몇 개 memo에서 반복 관찰됐는지
- `low_confidence_link_ratio`: confidence가 낮은 link 비율

주의: `node_coherence`는 relation tag overlap을 보는 근사 지표다.  
실제 제품 채택 판단에는 사람이 샘플을 보고 `맞는 node인지`, `너무 많이 붙은 것은 아닌지`, `link가 자기이해 카드로 쓸모 있는지`를 별도로 검수해야 한다.

---

# Eval Dataset으로 Semantic Graph 만들기

현재 eval 입력 파일은 `eval/` 아래에 있다.

```text
eval/
├── past_memos.json
├── past_memo_tags.json
├── eval_cases.json
└── eval_case_tags.json
```

## 1. 먼저 50개로 graph 생성 테스트

```bash
python3 memory_graph.py build-eval-graph \
  --state-dir data_eval \
  --force \
  --limit 50 \
  --data eval/past_memos.json \
  --past-tags eval/past_memo_tags.json \
  --summary-only
```

정상이라면 `data_eval/`에 상태 파일이 생성된다.

```text
data_eval/
├── memos.json
├── memo_annotations.json
├── memory_nodes.json
├── memory_evidence.json
└── memory_links.json
```

## 2. graph 생성 결과 확인

```bash
python3 - <<'PY'
import json
from pathlib import Path

base = Path("data_eval")

for name in [
    "memos.json",
    "memo_annotations.json",
    "memory_nodes.json",
    "memory_evidence.json",
    "memory_links.json",
]:
    path = base / name
    data = json.loads(path.read_text(encoding="utf-8"))
    print(name, len(data))

nodes = json.loads((base / "memory_nodes.json").read_text(encoding="utf-8"))
print("\nsample nodes:")
for node in list(nodes.values())[:5]:
    print("-", node["id"], node.get("title"), node.get("evidence_count"))
PY
```

기대 결과는 대략 아래와 같다.

```text
memos.json 50
memo_annotations.json 50
memory_nodes.json N
memory_evidence.json M
memory_links.json 0
```

`memory_nodes.json`이 0이면 LLM node judge가 동작하지 않은 것이다.  
`--skip-llm-node-judge`를 넣지 않았는지, Ollama와 모델 설정이 정상인지 확인한다.

## 3. 50개로 eval 실행

```bash
python3 memory_graph.py eval \
  --use-state \
  --state-dir data_eval \
  --data eval/past_memos.json \
  --past-tags eval/past_memo_tags.json \
  --eval-data eval/eval_cases.json \
  --eval-case-tags eval/eval_case_tags.json \
  --summary-only
```

상세 결과를 보고 싶으면 `--summary-only`를 제거한다.

```bash
python3 memory_graph.py eval \
  --use-state \
  --state-dir data_eval \
  --data eval/past_memos.json \
  --past-tags eval/past_memo_tags.json \
  --eval-data eval/eval_cases.json \
  --eval-case-tags eval/eval_case_tags.json
```

## 4. 500개 전체 graph 생성

50개 테스트가 성공하면 전체 500개를 대상으로 semantic graph를 만든다.

```bash
python3 memory_graph.py build-eval-graph \
  --state-dir data_eval \
  --force \
  --data eval/past_memos.json \
  --past-tags eval/past_memo_tags.json \
  --summary-only
```

500개 전체는 memo마다 LLM node judge를 호출하기 때문에 시간이 오래 걸릴 수 있다.

## 5. 500개 전체 eval

```bash
python3 memory_graph.py eval \
  --use-state \
  --state-dir data_eval \
  --data eval/past_memos.json \
  --past-tags eval/past_memo_tags.json \
  --eval-data eval/eval_cases.json \
  --eval-case-tags eval/eval_case_tags.json \
  --summary-only
```

---

# 결과 해석 방법

`eval` 출력은 dense baseline과 memory graph 결과를 함께 보여준다.

```text
[dense baseline]
...

[memory graph]
...
```

해석할 때는 아래를 같이 본다.

- memory graph가 dense baseline보다 hit rate를 올리는가
- zero-hit이 줄어드는가
- 특정 relation tag cluster에만 과도하게 유리하지 않은가
- memory graph가 dense 후보를 보완하는가, 아니면 좋은 dense 후보를 밀어내는가
- 생성된 memory node가 사람이 보기에도 자기이해 카드로 쓸 만한가

단순 hit rate가 높아도 memory node가 너무 넓거나 이상하면 제품 구조로는 부적합하다.  
반대로 hit rate가 조금 낮아도 node/evidence가 사람이 이해 가능한 자기이해 카드로 축적된다면 제품 방향에는 더 의미 있을 수 있다.

---

# Troubleshooting

## `memory_nodes.json`이 0개다

가능성이 높은 원인:

- `--skip-llm-node-judge`를 넣었다
- `--use-llm-node-judge` 없이 단건 ingest만 했다
- Ollama가 실행 중이 아니다
- `qwen3:8b` 같은 node judge model이 로컬에 없다
- `recall.generate_text()`가 JSON 응답을 제대로 받지 못했다

확인:

```bash
ollama list
ollama pull qwen3:8b
```

단건으로 먼저 확인한다.

```bash
python3 memory_graph.py ingest \
  --memo "초안을 쓰려다가 또 자료 조사만 하고 있다." \
  --relation-tags start_avoidance,overplanning \
  --use-llm-node-judge
```

## eval에서 memory graph 결과가 거의 dense와 같다

가능성이 높은 원인:

- 생성된 `memory_nodes`가 너무 적다
- `memory_evidence`가 거의 없다
- `eval_case_tags.json`의 relation tag가 너무 빈약하다
- `node_top_k`가 너무 낮거나 높다
- dense fallback 영향이 너무 크다

실험 옵션:

```bash
python3 memory_graph.py eval \
  --use-state \
  --state-dir data_eval \
  --data eval/past_memos.json \
  --past-tags eval/past_memo_tags.json \
  --eval-data eval/eval_cases.json \
  --eval-case-tags eval/eval_case_tags.json \
  --node-top-k 5 \
  --dense-top-k 50 \
  --summary-only
```

dense fallback 없이 순수 graph 효과를 보고 싶으면:

```bash
python3 memory_graph.py eval \
  --use-state \
  --state-dir data_eval \
  --data eval/past_memos.json \
  --past-tags eval/past_memo_tags.json \
  --eval-data eval/eval_cases.json \
  --eval-case-tags eval/eval_case_tags.json \
  --no-dense-fallback \
  --summary-only
```

## 기존 `data/`를 초기화하고 싶다

주의해서 실행한다.

```bash
rm -f data/memos.json \
      data/memo_annotations.json \
      data/memory_nodes.json \
      data/memory_evidence.json \
      data/memory_links.json
```

eval graph만 초기화하고 싶으면:

```bash
rm -rf data_eval
```

---

# 현재 코드 상태 요약

## `main.py`

`main.py`는 retrieval eval runner로 유지한다.

지원 기능:

- embedding 생성
- cosine similarity 기반 dense retrieval
- relation-only retrieval
- text + relation RRF
- relation candidate gate + text rerank
- dense candidate + LLM rerank
- eval 실행
- fixed relation tag 실험
- LLM relation tagging 실험

## `memory_graph.py`

`memory_graph.py`는 semantic memory graph PoC다.

지원 기능:

- 메모 저장
- relation tag annotation 저장
- dense evidence search
- relation tag overlap 기반 semantic node 후보 계산
- LLM node judge 기반 attach/create/ignore
- memory_evidence 누적
- memory_links 후보 생성
- semantic graph 기반 eval

현재 설계상 제거된 것:

- seed node
- `node_kind`
- relation tag를 node로 저장하는 방식
- fixture seed graph 기반 precision/recall

---

# 현재 판단

retrieval-only 실험은 baseline으로 유지한다.

하지만 KEO 제품 방향에서는 LLM reranker보다 memory graph가 더 적합하다.

이유:

- KEO의 핵심 연결은 같은 문장이 아니라 같은 반복 패턴이다
- 검색 결과를 매번 버리지 않고, memory node/evidence로 축적해야 한다
- relation tag는 최종 결과물이 아니라 node 후보를 좁히는 내부 신호에 가깝다
- LLM은 top-k 순서만 바꾸는 reranker보다, node를 만들고 evidence를 붙이는 maintainer 역할이 더 자연스럽다

따라서 다음 실험은 다음 질문으로 이어진다.

- semantic node가 너무 넓게 생기지 않는가
- 같은 패턴을 중복 node로 만들지 않는가
- node summary가 사용자에게 보여줘도 될 만큼 자연스러운가
- evidence가 충분히 설득력 있게 붙는가
- memory_links가 자기이해 카드 간 관계로 쓸모 있는가
