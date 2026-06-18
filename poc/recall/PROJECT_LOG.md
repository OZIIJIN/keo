# KEO Memory Graph — LLM 기반 개인 패턴 그래프 구축 기록

## 프로젝트 개요

개인 메모에서 반복 패턴을 자동으로 추출해 그래프로 연결하는 시스템.
사용자가 남긴 일상 메모들을 분석해 "메신저 확인 후 오전 흐름이 끊기는 패턴" 같은
구체적인 자기이해 카드(memory_node)를 생성하고, 누적된 evidence로 노드를 진화시키는 위키 패턴 구조.

### 핵심 설계 목표
- 모든 메모를 강제로 노드에 연결하는 게 아니라, **반복이 쌓일 때 자연스럽게 패턴이 보이는 구조**
- 노드 타이틀: 추상 카테고리 금지, "구체적 트리거 + 반복 결과" 구조 강제
- 위키 패턴: evidence 10개 이상 → evolve(split/refine/confirm) → 더 정확한 노드로 진화

---

## 아키텍처

```
memo 입력
→ relation_tags (사전 분류)
→ dense match (과거 메모 embedding 유사도 검색)
→ dense_node_hits (비슷한 메모들이 속한 노드 카운트)
→ candidate node 수집 (tag_overlap OR dense_count OR emb_score)
→ LLM judge: attach / create / ignore 판단
→ attach면 tag overlap guard 확인
→ evidence 추가
→ evidence 10개 이상이면 inline evolve 발동
→ post-build: evolve → merge 반복
```

**주요 컴포넌트**
- `process_memo_into_graph`: 메모 1개 처리 전체 파이프라인
- `collect_candidate_semantic_nodes`: LLM에 넘길 후보 노드 수집
- `judge_memory_node_with_llm`: attach/create/ignore LLM 판단
- `apply_node_judgement`: LLM 판단 적용 + evidence 추가
- `evolve_single_node`: evidence 누적된 노드 split/refine/confirm
- `merge_similar_nodes`: 유사 노드 embedding 기반 병합

---

## 빌드 버전별 결과

| 버전 | 메모 | active | pending | max_ev | avg_ev | 주요 변경 |
|------|------|--------|---------|--------|--------|-----------|
| llm_only | 500 | 76 | 0 | 165 | 13.6 | LLM judge만, evolve/merge 없음 |
| new2_100 | 100 | 10 | 28 | 17 | 10.9 | pending 시스템 + evolve 기초 |
| new7_100 | 100 | 24 | 22 | 16 | 6.4 | evolve split/refine/confirm 구현 |
| new8_100 | 100 | 31 | 22 | 17 | 5.6 | 30% rule, tags 3개 cap, post-build merge |
| new9_100 | 100 | 5 | 21 | 10 | 7.4 | 타이틀 "트리거+결과" 구조 강제 |
| new10_200 | 200 | 19 | 31 | 21 | 8.7 | tags cap 누락 버그 fix |
| new11_200 | 200 | 14 | 31 | 22 | 9.3 | merge while loop fix |
| new12_200 | 200 | 43 | 57 | 14 | 5.3 | trigger_keywords hard filter (실험) |

---

## 발견된 문제와 해결 과정

### 1. 과흡수 (Over-absorption)
**현상**: "맥락 분할로 인한 일정 방해" 같은 broad 노드가 evidence 17개+ 흡수.
관련 없는 메모들이 같은 relation_tag(context_fragmentation)를 공유한다는 이유로 계속 attach됨.

**근본 원인 분석**:
- attach 판단을 LLM에 전적으로 의존 → LLM은 "비슷한 결" 메모에 관대하게 attach
- candidate 수집 기준이 tag_overlap 하나만으로도 충분 → 너무 넓은 후보 풀
- `evidence_memo_ids` 벌크 attach: LLM이 attach 판단 시 지목한 과거 메모들까지 한꺼번에 추가되어 evidence count 뻥튀기

**시도한 해결**:
- 프롬프트: "The node's core trigger must be explicitly present in this memo" → 효과 제한적 (LLM이 여전히 관대)
- 타이틀 "트리거+결과" 구조 강제 → 타이틀 품질 개선됨, attach는 여전히 느슨
- trigger_keywords hard filter → 반대 문제: 파편화 (active 43개)

**남은 방향**:
1. tag-only candidate 제거: `if dense_count == 0 and emb_score == 0.0: continue`
2. representative evidence similarity guard: summary 말고 실제 evidence embedding과 비교
3. evidence_memo_ids attach에서 분리 (현재 memo만 붙이기)

### 2. LLM CONFIRM bias
**현상**: evolve가 발동해도 LLM이 계속 CONFIRM 반환. split이 거의 안 일어남.

**원인**: attach에서 관대하게 넣은 것을 evolve에서도 같은 LLM이 "다 비슷하네" 판단.
동일 LLM이 attach/evolve 양쪽에서 같은 방향으로 편향.

**해결**: evolve 프롬프트에 30% rule 추가 ("30% 이상이 트리거와 안 맞으면 반드시 split")

### 3. Summary boilerplate 복사
**현상**: split/refine 후 새 노드들의 summary가 원본 노드 summary를 그대로 복붙.
"이는 일정과 맥락 분할이 기억 상실을 유발하는 반복적 패턴이다"가 모든 split 노드에 등장.

**해결**: evolve 프롬프트에 "Write summary fresh — do NOT copy or paraphrase the original summary" 추가.

### 4. Split 결과 1개만 반환
**현상**: LLM이 split을 선택하고도 split_nodes를 1개만 반환. split 의미가 없어짐.

**해결**: `len(valid_candidates) < 2`이면 confirm으로 fallback.

### 5. split_nodes 타입 오류
**현상**: `AttributeError: 'str' object has no attribute 'get'`
LLM이 split_nodes 배열에 dict 대신 string 반환.

**해결**: `isinstance(sc, dict)` 체크 + 모든 필드 타입 검증 추가.

### 6. relation_tags 3개 cap 누락
**현상**: 노드 create 시 memo tags가 무제한 머지되어 tags 9개짜리 노드 생성.

**발견**: 두 곳에서 cap 누락
- `judge_memory_node_with_llm`의 `create_node["relation_tags"]` 머지 (line 636)
- `apply_node_evolution`의 split 후 remaining tags 재계산 (line 957)

**해결**: 두 곳 모두 `[:3]` 추가.

### 7. Merge 1-pass ordering miss
**현상**: post-build merge 후에도 동일 타이틀 노드 2개 잔존 (similarity 0.91).

**원인**: 단방향 순회(`indexed[i+1:]`)에서 node_a가 best match를 먼저 다른 노드와 merge해버리면, 실제 duplicate pair가 서로 못 만나는 체인 문제.

**해결**: `while merge_similar_nodes(...): pass` — 더 이상 merge 없을 때까지 반복.

### 8. Post-build evolve → merge 순서 문제
**현상**: active 14개 중 evolved가 1개뿐. 18-21ev 노드들이 evolve 안 됨.

**원인**: post-build 순서가 evolve → merge. evolve 시점에 이 노드들은 ec < 10이었으나, merge 후 다른 노드들이 흡수되면서 ec가 21, 18로 점프. 이미 evolve는 끝난 상태.

**방향**: merge 먼저 → evolve 순서로 변경, 또는 merge → evolve 반복.

---

## LLM 편향 패턴 관찰

| 판단 단계 | 관찰된 편향 |
|----------|------------|
| attach | 트리거 확인 없이 결과 유사성만으로 관대하게 attach |
| create | 메모 2-3개에서 추상화된 broad 제목 생성 (relation_tag 수준) |
| evolve/split | evidence 쌓인 노드를 "다 비슷한 패턴"으로 CONFIRM 선호 |
| evolve/refine | 원본 summary를 새 노드에 그대로 복붙 |

**핵심 인사이트**: attach와 evolve를 같은 LLM이 같은 bias로 판단하기 때문에,
attach에서 들어온 noise를 evolve에서도 걸러내지 못함.
LLM 판단 의존도를 줄이고 embedding 기반 구조적 가드를 강화해야 함.

---

## 평가 지표 설계

```python
# 빌드 후 자동 계산
active       # 실제 패턴으로 승격된 노드 수
pending      # 아직 반복 확인 안 된 후보 노드 수
max_ev       # 가장 많이 쌓인 노드의 evidence 수 (과흡수 지표)
avg_ev       # 평균 evidence (분산도 지표)
evolved      # evolve 실행된 노드 수
duplicates   # 동일 타이틀 노드 쌍 수
tags_over_3  # 태그 3개 초과 노드 수
```

**건강한 상태 기준** (200 메모 기준):
- active: 10-20개
- max_ev: 15 이하
- avg_ev: 5-9
- duplicates: 0
- tags_over_3: 0

---

## 기술 스택

- Python 3.10
- Ollama (로컬 LLM): qwen3:8b (judge), qwen3-embedding (embedding)
- Embedding 캐시: JSON 기반 (모델별 파일)
- 저장: JSON flat files (memory_nodes, memory_evidence, memos, memo_annotations)
