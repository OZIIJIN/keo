## PoC에서 증명하고 싶은 것
1. 메모 입력 -> 저장
2. 새 메모 입력 시 -> 과거 유사 메모 검색 (벡터 유사도)
3. 유사 메모들 기반 -> 패턴 + 분석 + 회고 생성

## 전체 흐름
1. 메모 입력
2. nomic-embed-text로 임베딩 생성
3. ChromaDB에 저장
4. 새 메모 입력 시 
5. 유사 메모 Top 3 검색
6. Ollama에 [새 메모 + 유사 메모들] 넘김
7. 패턴 + 분석 + 회고 생성

## 실행 순서 
1. 임베딩 모델 다운로드
```
ollama pull nomic-embed-text
```

2. 패키지 설치
```
cd poc/recall
pip install -r requirements.txt
```

3. 서버 실행
```
uvicorn main:app --reload
```