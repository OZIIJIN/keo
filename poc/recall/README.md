## 목표

`poc/recall`은 KEO recall retrieval의 baseline과 보강 신호를 검증하는 실험 디렉토리다.

현재 이 PoC에서 보고 있는 질문은 아래다.
- 한국어 메모에서 text dense retrieval이 기본적으로 어느 정도 동작하는가
- relation tag가 retrieval signal로 실제 도움이 되는가
- 원문 의미 채널과 relation 채널을 분리했을 때 성능이 유지되거나 좋아지는가
- KEO다운 "같은 사건"이 아니라 "같은 패턴 역할" 연결을 eval이 제대로 측정하고 있는가

## 현재 구조

현재 흐름은 세 채널 실험으로 정리되어 있다.

- `text`
  - 메모 원문만 embed해서 retrieval
- `relation`
  - relation tag만 embed해서 retrieval
- `rrf`
  - text ranking과 relation ranking을 따로 만든 뒤 RRF로 fuse
- `relation_gate`
  - relation retrieval로 후보군을 먼저 줄인 뒤 text dense로 rerank

즉 지금은 `원문 + 태그를 한 문자열로 합쳐 한 번에 embed`하는 방식보다, 채널을 분리해서 다루는 쪽을 우선 본다.

## 현재 결론

현재까지 확인한 방향은 이렇다.

- dense-only baseline은 여전히 필요하다
- relation tag는 retrieval signal로 실제 도움이 된다
- `text + relation RRF`가 현재 가장 안정적인 보강 구조다
- eval miss 중 일부는 retrieval 실패보다 `expected_ids` undercoverage 문제였다

핵심은 KEO retrieval이 단순 문장 유사도보다 `반복 패턴`, `반응 방식`, `회고 역할` 연결에 더 가깝다는 점이다.

## 현재 결과 스냅샷

기준
- dataset: `past_memos.json` 215개, `eval_cases.json` 106개
- top-k: 10
- embedding model: `qwen3-embedding`
- relation annotation: fixed

결과

| mode | hits | expected | hit rate | zero-hit |
| --- | ---: | ---: | ---: | ---: |
| `text` | 134 | 479 | 27.97% | 22 |
| `relation` | 159 | 479 | 33.19% | 16 |
| `rrf` | 160 | 479 | 33.40% | 20 |
| `relation_gate` (`candidate_k=30`) | 153 | 479 | 31.94% | 17 |

현재 해석
- relation signal은 실제 gain을 만든다
- `primary/secondary` 구분보다 `relation_tags` 자체가 더 중요했다
- `rrf`는 relation gain을 유지하면서 text 채널을 분리해둘 수 있어서 구조적으로 가장 낫다
- `relation_gate`는 가능성은 있지만 `candidate_k`에 민감하다

## 데이터셋

- 과거 메모: [eval/past_memos.json](/Users/vonai/Desktop/keo/poc/recall/eval/past_memos.json)
- 평가 케이스: [eval/eval_cases.json](/Users/vonai/Desktop/keo/poc/recall/eval/eval_cases.json)
- relation tag 스키마: [tag_schema.json](/Users/vonai/Desktop/keo/poc/recall/tag_schema.json)
- 과거 메모 태그: [eval/past_memo_tags.json](/Users/vonai/Desktop/keo/poc/recall/eval/past_memo_tags.json)
- 평가 케이스 태그: [eval/eval_case_tags.json](/Users/vonai/Desktop/keo/poc/recall/eval/eval_case_tags.json)

현재 데이터는 KEO 스타일의 메모를 더 다양하게 포함한다.

- 짧은 자기관찰 메모
- 생활 행정 일 미룸 메모
- 일정 전후 한 줄 메모
- 책/문장/인용이 오래 남는 메모
- 제품 아이디어와 일상 메모가 같은 흐름에 섞이는 메모
- 에너지 저하, 리듬 붕괴, 답장 미룸, 예약 미룸 같은 생활성 메모

의도적으로 너무 템플릿처럼 보이지 않게, 실제 사람이 그때그때 남겼을 법한 질감으로 섞어뒀다.

## Relation Retrieval용 Eval 원칙

relation retrieval에서는 `expected_ids`를 좁은 surface match 정답셋으로 잡으면 안 된다. 현재 eval은 아래 원칙을 따른다.

### 1. 같은 표현보다 같은 패턴 역할을 우선한다

정답은 "같은 단어를 쓴 메모"가 아니라 아래 중 하나면 포함 가능하다.

- 같은 반응 방식
- 같은 판단 흐름
- 같은 회고 역할
- 같은 여파를 설명하는 메모

예를 들어:
- 일정 전후 메모를 같이 봐야 패턴이 보인다는 query는
  - 회의 전 기대
  - 회의 후 한 줄
  - 일정 직후 남긴 짧은 정리
  를 모두 정답 후보로 볼 수 있다.

### 2. 같은 memo type이 아니어도 정답이 될 수 있다

KEO에서는 아래처럼 서로 다른 표면 타입도 같은 역할이면 연결 대상이 된다.

- quote 메모 <-> 그 문장이 하루 판단에 남은 메모
- 생활 일 미룸 메모 <-> 그 미룸이 자존감에 남는 메모
- 제품 아이디어 메모 <-> 자기이해 구조를 제품으로 옮기려는 메모

즉 `memo type` 일치보다 `회고/패턴 역할` 일치를 더 우선한다.

### 3. expected_ids는 한 cluster의 대표 예시만 두지 않는다

기존 miss의 큰 원인 중 하나는 정답셋이 특정 하위 표현군에만 묶여 있던 점이었다.

그래서 relation eval에서는:
- 같은 cluster 안의 다른 자연스러운 변형 메모도 포함
- 너무 특정 표현에만 편향된 정답셋은 확장
- 한 패턴을 설명하는 생활형/회고형/인용형 메모를 함께 허용

### 4. zero-hit을 줄이는 방향을 정답셋도 반영한다

관계가 명확한 후보가 top-k에 들어왔는데 `expected_ids`에 없어서 miss가 되는 상황을 줄여야 한다.

따라서 expected 확장 audit에서는 아래를 중점적으로 본다.
- retrieval top-k에 반복해서 뜨는 메모인가
- 사람이 읽어도 query와 역할상 충분히 가깝나
- 기존 expected가 지나치게 보수적이진 않았나

### 5. relation eval은 완전 동일문장 검색이 아니다

이 eval의 목적은 "문장이 똑같은 메모 찾기"가 아니다.

목적은:
- 같은 미룸 패턴
- 같은 일정 여파
- 같은 문장 여운
- 같은 자기속임 구조
- 같은 생활성 unfinished loop

를 복원하는 retrieval이 가능한지 보는 것이다.

## 이번 audit에서 확장한 케이스

특히 아래 케이스들은 `expected_ids`를 확장했다.

- `case-028`
  - 일정 전후 메모와 짧은 회고 연결 cluster를 더 넓게 포함
- `case-037`, `case-039`, `case-041`
  - 반품, 주문, 예약, 세탁세제 같은 생활성 미룸 cluster 확장
- `case-052` ~ `case-055`
  - quote 자체뿐 아니라 문장 여운, 판단 여파, 패턴 설명 메모까지 포함

즉 이번 audit의 핵심은 `같은 표현군만 정답으로 인정하던 좁은 정답셋`을 `같은 패턴 역할 메모`까지 포함하도록 넓힌 것이다.

## 구현 원칙

- 가장 작은 동작 가능한 실험 구조 유지
- retrieval 품질과 generation 품질을 분리해서 보기
- 로컬에서 바로 재현 가능해야 함
- 실험 결과를 사람이 읽기 쉽게 출력
- eval dataset을 계속 보정 가능한 구조로 유지

## 현재 코드 상태

현재 [main.py](/Users/vonai/Desktop/keo/poc/recall/main.py)는 아래를 지원한다.

- embedding 생성
- cosine similarity 기반 dense retrieval
- relation-only retrieval
- text + relation RRF
- relation candidate gate + text rerank
- eval 실행
- fixed relation tag 실험
- LLM relation tagging 실험

현재 빠진 것
- sparse retrieval
- BM25 hybrid
- dedicated reranker model
- relation embedding과 text embedding의 가중 fusion

## 실행 방법

전제
- Ollama가 실행 중이어야 한다
- 사용할 embedding / tagging 모델이 로컬에 있어야 한다

기본 text-only 검색

```bash
cd poc/recall
ollama pull qwen3-embedding
python3 main.py \
  --memo "회의와 알림에 계속 끌려다녀서 중요한 작업을 제대로 못 했다." \
  --model qwen3-embedding \
  --retrieval-mode text \
  --top-k 10
```

relation-only 단건 검색

```bash
python3 main.py \
  --memo "완벽하게 하려다가 또 시작을 미루는 중이다. 일단 거칠게라도 초안부터 만들어야 할 듯." \
  --model qwen3-embedding \
  --retrieval-mode relation \
  --relation-tags start_avoidance,evaluation_fear
```

RRF 단건 검색

```bash
python3 main.py \
  --memo "완벽하게 하려다가 또 시작을 미루는 중이다. 일단 거칠게라도 초안부터 만들어야 할 듯." \
  --model qwen3-embedding \
  --retrieval-mode rrf \
  --relation-tags start_avoidance,evaluation_fear
```

LLM으로 relation tag를 뽑아 단건 검색

```bash
python3 main.py \
  --memo "완벽하게 하려다가 또 시작을 미루는 중이다. 일단 거칠게라도 초안부터 만들어야 할 듯." \
  --model qwen3-embedding \
  --retrieval-mode rrf \
  --tagging-mode llm
```

eval 전체 실행

```bash
python3 main.py \
  --eval \
  --model qwen3-embedding \
  --retrieval-mode text \
  --top-k 10
```

relation-only eval

```bash
python3 main.py \
  --eval \
  --model qwen3-embedding \
  --retrieval-mode relation \
  --tagging-mode fixed \
  --top-k 10
```

RRF eval

```bash
python3 main.py \
  --eval \
  --model qwen3-embedding \
  --retrieval-mode rrf \
  --tagging-mode fixed \
  --top-k 10
```

relation gate eval

```bash
python3 main.py \
  --eval \
  --model qwen3-embedding \
  --retrieval-mode relation_gate \
  --relation-candidate-k 30 \
  --tagging-mode fixed \
  --top-k 10
```

summary만 보기

```bash
python3 main.py \
  --eval \
  --model qwen3-embedding \
  --retrieval-mode rrf \
  --tagging-mode fixed \
  --top-k 10 \
  --summary-only
```

## LLM tagging 메모

- 기본 tagging model은 `qwen3:8b`
- 현재 relation tagging은 `primary/secondary`가 아니라 `relation_tags`만 생성한다
- 현재 병목은 retrieval core보다 `LLM relation tagging latency + tag 경계 안정성` 쪽에 더 가깝다

즉 실험 질문은 점점 더 명확해졌다.

- fixed relation tag는 retrieval signal로 유효한가: `예`
- LLM이 그 relation signal을 자동으로 충분히 잘 복원하는가: 아직 추가 검증 필요
