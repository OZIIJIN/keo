## 목표

`poc/recall`은 KEO recall의 retrieval baseline을 검증하기 위한 실험 디렉토리다.

현재 기준 baseline은 dense-only retrieval이다.

이 PoC에서 확인하려는 것은 아래다.
- 한국어 메모에서 embedding retrieval이 기본적으로 동작하는가
- KEO 스타일 memo dataset에서 의미 기반 연결이 어느 정도 되는가
- 어떤 embedding 모델이 현재 로컬 PoC 환경에서 더 유리한가
- dense retrieval을 baseline으로 둘 만큼의 최소 품질이 나오는가

## 현재 결론

현재 기준으로는 `dense-only + qwen3-embedding`이 baseline으로 가장 적합했다.

즉, 지금 `poc/recall`은
- hybrid retrieval 실험을 한 번 거쳤고
- 그 결과 BM25 hybrid는 보류하고
- dense-only baseline을 유지하는 상태다

## 현재 목표 흐름

현재 메모 입력
-> embedding 생성
-> 과거 메모 dense retrieval
-> top-k 출력

## 범위

포함
- 현재 메모 입력
- 과거 메모 저장
- embedding 생성
- cosine similarity 기반 top-k retrieval
- eval dataset 일괄 실행
- 모델 교체 실험

제외
- BM25 / sparse retrieval
- hybrid fusion
- reranker
- query expansion
- 회고 문장 생성
- UI 연결

## 1차 PoC 결과

1차 PoC에서는 dense-only baseline과 embedding 모델 비교를 했다.

조건
- `past_memos.json`: 200개
- `eval_cases.json`: 100개
- top-k: 10
- cosine similarity

결과

| model | hits | expected | hit rate | zero-hit |
| --- | ---: | ---: | ---: | ---: |
| `nomic-embed-text` | 55 | 479 | 11.48% | 62 |
| `bge-m3` | 121 | 479 | 25.26% | 30 |
| `bona/bge-m3-korean` | 122 | 479 | 25.47% | 33 |
| `nomic-embed-text-v2-moe` | 122 | 479 | 25.47% | 29 |
| `qllama/multilingual-e5-large` | 127 | 479 | 26.51% | 29 |
| `qwen3-embedding` | 134 | 479 | 27.97% | 22 |

1차 결론
- `nomic-embed-text`는 한국어 memo retrieval baseline으로 약했다
- `qwen3-embedding`이 현재 dense-only 최고 후보였다
- dense-only가 충분히 높다고 보긴 어렵지만, baseline으로는 가장 나았다

## 2차 PoC 결과

2차 PoC에서는 dense-only 한계를 넘기기 위해
BM25 기반 sparse retrieval + RRF fusion hybrid를 시도했다.

실험한 것
- 정규식 기반 한국어 토큰화
- 조사 보정 규칙 추가
- `pecab` 기반 토큰화
- `candidate_k`, `rrf_k` 조정

핵심 비교
- dense-only baseline: `134 / 479`
- 첫 hybrid baseline: `122 / 479`
- 가장 잘 나온 hybrid: `131 / 479`

2차 결론
- 현재 dataset 기준으로 BM25 hybrid는 dense-only를 넘지 못했다
- sparse signal이 dense를 보완하기보다 노이즈를 넣는 경우가 더 많았다
- `candidate_k` 조정으로 일부 개선은 있었지만 baseline 역전에는 실패했다
- 따라서 현재 baseline은 hybrid가 아니라 dense-only로 유지한다

## 왜 BM25 hybrid를 보류했는가

- KEO memo retrieval은 단순 키워드 overlap보다 의미 연결 비중이 크다
- 한국어 tokenization과 sparse 설계 비용이 생각보다 컸다
- 현재 BM25 품질로는 dense top-k를 흔들고도 성능 이득이 나지 않았다

즉, 문제는 hybrid라는 아이디어 자체보다
현재 BM25 sparse signal이 KEO memo retrieval에 충분히 맞지 않았다는 쪽에 가깝다.

## 다음 단계 후보

다음 PoC는 아래 중 하나로 가는 편이 더 유리하다.

- dense-only 유지 + reranker
- dense-only 유지 + query expansion
- dense retrieval 위에 ontology layer를 얹는 방식

즉, 다음 개선은 sparse retrieval 확장보다
dense baseline 위 보강 레이어 쪽을 우선 검토한다.

## 구현 원칙

- 가장 작은 동작 가능한 baseline 유지
- retrieval 품질을 generation과 분리해서 검증
- 로컬에서 바로 재현 가능해야 함
- 실험 결과를 사람이 읽기 쉽게 출력
- eval dataset으로 반복 비교 가능해야 함

## 현재 코드 상태

현재 [main.py](/Users/vonai/Desktop/keo/poc/recall/main.py)는 dense-only baseline 코드다.

현재 들어있는 것
- embedding 생성
- cosine similarity 기반 dense retrieval
- eval 실행
- 모델 교체 실험
- embedding 캐시

현재 빠진 것
- BM25 retrieval
- hybrid fusion
- reranker
- LLM 기반 query expansion

현재 추가된 것
- 고정 relation tag
- relation tag를 retrieval text에 붙이는 expansion mode
- query/eval case에 대한 LLM 태깅

## 실행 방법

전제
- Ollama가 실행 중이어야 한다
- 사용할 embedding 모델이 로컬에 있어야 한다

기본 실행

```bash
cd poc/recall
ollama pull qwen3-embedding
python3 main.py --memo "회의와 알림에 계속 끌려다녀서 중요한 작업을 제대로 못 했다." --model qwen3-embedding --top-k 10
```

태그를 붙여 단건 검색

```bash
python3 main.py \
  --memo "완벽하게 하려다가 또 시작을 미루는 중이다. 일단 거칠게라도 초안부터 만들어야 할 듯." \
  --model qwen3-embedding \
  --expansion-mode query_only \
  --relation-tags start_avoidance,evaluation_fear
```

LLM으로 단건 query 태깅 후 검색

```bash
python3 main.py \
  --memo "완벽하게 하려다가 또 시작을 미루는 중이다. 일단 거칠게라도 초안부터 만들어야 할 듯." \
  --model qwen3-embedding \
  --expansion-mode query_only \
  --tagging-mode llm
```

eval 전체 실행

```bash
python3 main.py --eval --model qwen3-embedding --top-k 10
```

relation tag만 붙인 `both` 실험

```bash
python3 main.py \
  --eval \
  --model qwen3-embedding \
  --top-k 10 \
  --expansion-mode both \
  --tagging-mode fixed
```

LLM relation tag로 `both` 실험

```bash
python3 main.py \
  --eval \
  --model qwen3-embedding \
  --top-k 10 \
  --expansion-mode both \
  --tagging-mode llm
```

summary만 보기

```bash
python3 main.py --eval --model qwen3-embedding --top-k 10 --summary-only
```

## 데이터

- 과거 메모: [eval/past_memos.json](/Users/vonai/Desktop/keo/poc/recall/eval/past_memos.json)
- 평가 케이스: [eval/eval_cases.json](/Users/vonai/Desktop/keo/poc/recall/eval/eval_cases.json)
- 태그 스키마: [tag_schema.json](/Users/vonai/Desktop/keo/poc/recall/tag_schema.json)
- 과거 메모 태그: [eval/past_memo_tags.json](/Users/vonai/Desktop/keo/poc/recall/eval/past_memo_tags.json)
- 평가 케이스 태그: [eval/eval_case_tags.json](/Users/vonai/Desktop/keo/poc/recall/eval/eval_case_tags.json)

평가 케이스의 `expected_ids`는 단일 정답이 아니다.
top-k 안에 들어오면 좋은 관련 후보 메모 집합이다.

## 태그 기반 retrieval

태그 구조
- relation tag는 메모가 가진 반복 신호를 나타낸다
- 현재 retrieval 실험은 relation tag만 사용한다

expansion mode
- `none`: 기존 dense-only baseline
- `query_only`: 현재 메모나 eval case query에만 태그를 붙여 embed
- `both`: 현재 메모와 과거 메모 모두 태그를 붙여 embed

tagging mode
- `fixed`: 단건 검색은 CLI 입력 relation tag를 쓰고, eval은 `eval_case_tags.json`을 사용
- `llm`: 현재 메모나 eval case를 Ollama generation model로 relation tagging

기본 tagging model
- `qwen3:8b`
- 필요하면 `--tagging-model`로 덮어쓸 수 있다

annotation 파일 규격

```json
{
  "memo-001": {
    "relation_tags": ["event_reflection_link", "self_understanding_product"]
  }
}
```
