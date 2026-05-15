---
name: keo-recall-eval-dataset
description: Generate KEO-style retrieval evaluation datasets for the recall POC by creating `past_memos.json` and `eval_cases.json` with realistic multi-day memo flows, mixed memo types, and semantically grounded expected matches. Use when Codex needs to create or refresh eval data for memo retrieval experiments in `poc/recall`, especially when the dataset must reflect KEO's self-understanding product concept rather than a generic memo app.
---

# KEO Recall Eval Dataset

KEO recall POC용 eval dataset을 생성하거나 확장한다. 이 skill은 사용자 요청에 맞는 데이터를 직접 만들고, 번들 스크립트로 검증해서 저장하는 용도다.

## 언제 쓰는가

다음처럼 `poc/recall`용 retrieval 평가 데이터를 만들거나 갱신할 때 사용한다.

- `past_memos.json`이 필요할 때
- `eval_cases.json`이 필요할 때
- KEO 컨셉에 맞는 메모 분포가 필요할 때
- 키워드보다 의미 유사성이 중요한 retrieval case가 필요할 때
- 며칠 동안 한 사람이 남긴 흐름처럼 보이는 메모 데이터가 필요할 때
- 특정 테마나 편향을 가진 추가 데이터셋을 만들고 싶을 때
- 기존 eval에 새 배치를 append하고 싶을 때

## KEO 메모 데이터의 성격

데이터는 일반적인 메모앱 테스트셋이 아니라 KEO의 자기이해 컨셉에 맞아야 한다.

- 메모는 단순 저장물이 아니라, 나중에 다시 연결되고 해석될 재료로 본다.
- 기억, 실행, 회고, 계획, 제품 아이디어가 한 흐름 안에 섞여 있어야 한다.
- 메모는 실제 사용자가 급하게 남긴 것처럼 자연스러워야 한다.
- 문장은 약간 거칠고, 축약되거나, 덜 정리되어 있어도 된다.
- 일정, 행동 실패, 감정, 자기 해석, 이후 회고가 서로 이어질 수 있어야 한다.
- 표면 키워드 유사성과 더불어 반복 패턴이나 맥락 유사성도 살아 있어야 한다.

메모 분포에는 아래 타입이 섞여 있어야 한다.

- 일상적인 짧은 메모
- 제품 아이디어 메모
- 계획 메모
- 책이나 문장 기록
- 자기 해석 메모
- 실행 실패, 회피, 우선순위 혼란 관련 메모
- 일정과 회고의 연결 가능성을 보여주는 메모

## 작업 흐름

1. 사용자 요청을 읽고 어떤 데이터셋이 필요한지 먼저 정한다.

- 새로 만드는지, 기존 eval에 이어 붙이는지 판단한다.
- 메모 길이, 기간, 톤, 메모 타입 분포, retrieval 난이도를 정한다.
- 항상 generic memo app이 아니라 KEO 관점으로 해석한다.

2. 요청에 맞는 dataset 내용을 직접 생성한다.

- `past_memos`와 `eval_cases`를 요청에 맞춰 작성한다.
- 미리 고정된 샘플을 그대로 재사용하지 않는다.
- 하나의 사람이 연속된 날짜에 남긴 기록처럼 보이게 만든다.

3. 생성한 결과를 검증하고 저장한다.

- 먼저 아래 형태의 JSON을 만든다.

```json
{
  "past_memos": [
    {
      "id": "memo-001",
      "date": "2026-04-07",
      "text": "소화기 앱 -> pds 다이어리 방향도 괜찮을까. 3개월 회고, 하루 회고 같이."
    }
  ],
  "eval_cases": [
    {
      "id": "case-001",
      "current_memo": "완벽하게 준비하려다가 시작을 계속 미루고 있다. 일단 초안부터 만들어야 할 것 같다.",
      "expected_ids": ["memo-001", "memo-002"]
    }
  ]
}
```

- 새로 덮어쓸 때:

```bash
python3 skills/keo-recall-eval-dataset/scripts/materialize_eval_dataset.py \
  --input /path/to/dataset.json \
  --output-dir ./eval \
  --force
```

- 기존 eval에 append할 때:

```bash
python3 skills/keo-recall-eval-dataset/scripts/materialize_eval_dataset.py \
  --input /path/to/dataset.json \
  --output-dir ./eval \
  --merge
```

스크립트는 schema와 ID를 검증한 뒤 아래 파일을 쓴다.

- `eval/past_memos.json`
- `eval/eval_cases.json`

`--merge`를 쓸 때는 아래를 지킨다.

- 새 배치의 ID는 기존 `memo-*`, `case-*`와 겹치지 않게 만든다.
- 입력 JSON은 전체 교체본이 아니라 추가 배치로 취급한다.
- 새 `expected_ids`는 merge 후 존재하는 메모를 가리켜야 한다.
- 새 메모뿐 아니라 기존 `eval/past_memos.json`의 메모를 참조해도 된다.

## 생성 규칙

- 메모는 여러 날에 걸친 하나의 흐름으로 만든다.
- 화자 톤은 일관되어야 한다.
- 문장을 너무 매끈하게 다듬지 않는다.
- 일부 메모는 짧고 맥락 의존적이어도 괜찮다.
- 쉬운 retrieval case와 애매한 case를 섞는다.
- `expected_ids`는 의미 유사성, 반복 행동 패턴, 회고 연결성, 결정 맥락 기준으로 고른다.
- 일부 case는 제품 메모와 연결되고, 일부는 행동 패턴과 연결되고, 일부는 일정-회고 연결을 보여줘야 한다.
- 사용자가 특정 테마의 추가 배치를 원하면 그 요청에 맞는 새 내용을 만든다.
- append할 때는 기존 ID를 재사용하지 말고 자연스럽게 다음 번호를 이어간다.
- 사용자가 랜덤 생성을 명시하지 않는 한 결과는 재현 가능하게 유지한다.

## 금지사항

- 일반적인 생산성 앱 테스트 데이터처럼 만들지 않는다.
- 감정 일기만 가득한 데이터셋으로 만들지 않는다.
- 모든 메모를 교과서처럼 완결된 문장으로 쓰지 않는다.
- 같은 키워드가 나온다는 이유만으로 `expected_ids`를 고르지 않는다.
- 모든 case를 너무 쉽게 만들지 않는다.
- 사용자가 원하지 않았는데 설명용 필드를 추가하지 않는다.
- 하나의 고정 샘플 데이터셋만 반복 출력하는 구조로 두지 않는다.
- merge 시 기존 데이터를 몰래 재번호 붙이지 않는다.
- 사용자가 원하지 않았는데 기존 결과를 덮어쓰지 않는다.

## 출력 형식

`past_memos.json`

```json
[
  {
    "id": "memo-001",
    "date": "2026-04-07",
    "text": "소화기 앱 -> pds 다이어리 방향도 괜찮을까. 3개월 회고, 하루 회고 같이."
  }
]
```

`eval_cases.json`

```json
[
  {
    "id": "case-001",
    "current_memo": "완벽하게 준비하려다가 시작을 계속 미루고 있다. 일단 초안부터 만들어야 할 것 같다.",
    "expected_ids": ["memo-001", "memo-002"]
  }
]
```

## 품질 체크리스트

작업을 끝내기 전에 아래를 확인한다.

- 데이터셋이 현재 사용자 요청을 반영하는가
- 한 사람이 여러 날 남긴 흐름처럼 보이는가
- 필요한 메모 타입이 골고루 섞였는가
- 문장 톤이 너무 교과서적이지 않고 자연스러운가
- 최소 일부 case는 키워드가 아니라 의미 기반 retrieval을 요구하는가
- 모든 `expected_id`가 실제 `past_memos.json`에 존재하는가
- memo ID와 case ID에 중복이 없는가
- append 배치가 기존 eval ID와 충돌하지 않는가
- 결과 JSON이 유효하고 읽기 좋게 저장되는가
- 결과 파일이 기본적으로 `poc/recall/eval` 아래에 들어가는가
