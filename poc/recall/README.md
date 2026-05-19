## 목표

`poc/recall`은 KEO recall의 hybrid retrieval PoC를 검증하기 위한 실험 디렉토리다.

이번 단계에서 확인하려는 것은 아래다.
- 한국어 메모에서 dense retrieval과 sparse retrieval을 함께 쓰면 품질이 실제로 좋아지는가
- 현재 메모가 들어왔을 때 "지금 다시 볼 만한 과거 기록"을 더 안정적으로 가져올 수 있는가
- 가장 단순한 hybrid 구조만으로도 dense-only 대비 zero-hit를 줄일 수 있는가
- 로컬에서 반복 실험 가능한 최소 구조를 만들 수 있는가

## 현재 목표 구조

현재 PoC의 목표 흐름은 아래다.

현재 메모 입력
-> dense retrieval
-> sparse retrieval
-> candidate union
-> fusion score 정렬
-> final top-k 출력

초기 구현 방향
- dense: `qwen3-embedding`
- sparse: BM25 또는 가장 단순한 keyword search
- fusion: RRF 우선

즉, 이 디렉토리의 현재 목표는
"좋은 embedding 하나를 찾는 것"이 아니라
"dense-only 한계를 넘기 위한 가장 작은 hybrid retrieval baseline"을 만드는 것이다.

## 범위

포함
- 현재 메모 입력
- 과거 메모 저장
- dense 후보 검색
- sparse 후보 검색
- 후보 합치기
- fusion 기반 재정렬
- eval dataset 일괄 실행
- top-k 결과 출력

제외
- 회고 문장 생성
- 패턴 분석 고도화
- 실행 엔진 연결
- 모바일 UI 연결
- 운영 구조 / 서비스 구조

## 왜 hybrid로 가는가

1차 dense-only PoC에서 확인한 결론은 간단했다.
- embedding 모델 차이는 분명히 있었다
- `qwen3-embedding`이 가장 나은 dense 후보였다
- 그래도 dense-only만으로는 KEO가 원하는 recall 품질이 부족했다

dense-only benchmark 요약
- eval cases: 100
- expected total: 479
- best dense model: `qwen3-embedding`
- hits: `134 / 479`
- hit rate: `27.97%`
- zero-hit: `22`

이 결과는
"dense 모델을 더 고르면 해결된다"보다
"retrieval 구조 자체를 hybrid로 바꿔야 한다"는 쪽에 더 가깝다.

## 좋은 retrieval 기준

- 단순 키워드 중복이 아니라 의미나 맥락이 연결됨
- 같은 주제, 고민, 행동 패턴, 과거 다짐이 후보에 포함됨
- 현재 생각을 확장하는 과거 기록이 포함됨
- 엉뚱한 결과가 줄어듦
- 사람이 봤을 때 "지금 다시 볼 만한 과거 기록"이라고 느껴짐

## 평가 질문

1. hybrid top-k 안에 expected memo가 들어오는가
2. dense-only보다 zero-hit가 줄어드는가
3. dense-only에서 놓치던 키워드성 연결을 sparse가 보완하는가
4. sparse-only가 놓치던 의미 연결을 dense가 보완하는가
5. 최종 fusion 결과가 사람이 보기에 더 자연스러운가

## 구현 원칙

- 가장 작은 동작 가능한 hybrid 흐름부터 구현
- dense, sparse, fusion을 분리해서 검증 가능하게 만들기
- 로컬에서 바로 재현 가능해야 함
- 실험 결과를 사람이 읽기 쉬워야 함
- 데이터와 평가 기준을 파일로 남겨 반복 비교 가능해야 함

## 현재 코드 상태

현재 [main.py](/Users/vonai/Desktop/keo/poc/recall/main.py)는 최소 hybrid baseline까지 포함한다.

현재 들어있는 것
- embedding 생성
- cosine similarity 기반 dense retrieval
- BM25 기반 sparse retrieval
- candidate union
- RRF 기반 fusion
- eval 실행
- 모델 교체 실험
- embedding 캐시

아직 추가해야 하는 것
- sparse 품질 개선용 한국어 토큰화 고도화
- fusion 가중치 실험
- hybrid 지표 비교 자동화 확장

즉, 현재는 가장 작은 hybrid baseline이 들어갔고,
이제 품질 개선 실험을 이어가면 된다.

## 실행 방법

전제
- Ollama가 실행 중이어야 한다
- dense retrieval에 사용할 embedding 모델이 로컬에 있어야 한다
- Python 의존성을 먼저 설치해야 한다

현재 hybrid baseline을 실행하는 방법

```bash
cd poc/recall
python3 -m pip install -r requirements.txt
ollama pull qwen3-embedding
python3 main.py --memo "회의와 알림에 계속 끌려다녀서 중요한 작업을 제대로 못 했다." --model qwen3-embedding --top-k 10
```

eval 전체 실행

```bash
python3 main.py --eval --model qwen3-embedding --top-k 10
```

summary만 보기

```bash
python3 main.py --eval --model qwen3-embedding --top-k 10 --summary-only
```

dense-only와 비교하고 싶으면

```bash
python3 main.py --eval --mode dense --model qwen3-embedding --top-k 10 --summary-only
```

## 데이터

- 과거 메모: [eval/past_memos.json](/Users/vonai/Desktop/keo/poc/recall/eval/past_memos.json)
- 평가 케이스: [eval/eval_cases.json](/Users/vonai/Desktop/keo/poc/recall/eval/eval_cases.json)

평가 케이스의 `expected_ids`는 단일 정답이 아니다.
top-k 안에 들어오면 좋은 관련 후보 메모 집합이다.
