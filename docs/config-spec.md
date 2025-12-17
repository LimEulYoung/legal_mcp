# Configuration Specification

Legal Search MCP 서버의 설정 파일(`config.py`) 명세서

---

## 📋 개요

RRF 하이브리드 검색의 모든 파라미터를 중앙에서 관리하는 설정 모듈

**파일 위치:** `src/config.py`

---

## 🔧 환경 변수 설정

### 필수 환경 변수

`.env` 파일에 다음 값들이 필수로 설정되어야 합니다:

```env
# Upstage API
UPSTAGE_API_KEY=up_xxxxxxxxxxxxx

# Elasticsearch
ES_HOST=3.36.103.251
ES_PORT=9200
ES_SCHEME=http
ES_USER=elastic
ES_PASSWORD=+IJgf+zk2HjKA*n5lQPP
```

---

## ⚙️ 설정 항목

### 1. Elasticsearch 연결 설정

| 변수명 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `ES_HOST` | str | 환경변수 | Elasticsearch 호스트 주소 |
| `ES_PORT` | int | 9200 | Elasticsearch 포트 |
| `ES_SCHEME` | str | "http" | 프로토콜 (http/https) |
| `ES_USER` | str | 환경변수 | 인증 사용자명 |
| `ES_PASSWORD` | str | 환경변수 | 인증 비밀번호 |

---

### 2. 인덱스명

| 변수명 | 타입 | 값 | 설명 |
|--------|------|-----|------|
| `INDEX_COURT_CASES` | str | "test_court_cases_new" | 판례 인덱스명 |
| `INDEX_STATUTES_METADATA` | str | "test_statutes_metadata" | 법령 메타데이터 인덱스명 |
| `INDEX_STATUTES` | str | "test_statutes" | 법령 조문 인덱스명 |

---

### 3. LLM 및 임베딩 설정

| 변수명 | 타입 | 값 | 설명 |
|--------|------|-----|------|
| `UPSTAGE_API_KEY` | str | 환경변수 | Upstage API 인증 키 |
| `EMBEDDING_MODEL` | str | "embedding-query" | 임베딩 모델명 |
| `EMBEDDING_DIMENSIONS` | int | 4096 | 임베딩 벡터 차원 |

---

### 4. 기본 검색 설정

| 변수명 | 타입 | 값 | 설명 |
|--------|------|-----|------|
| `SEARCH_TOP_K` | int | 5 | 검색 결과 개수 (판례/법령 공통) |
| `CASE_SUMMARY_MAX_LENGTH` | int | 1500 | 판례 검색 시 judgment_summary 최대 글자 수 |
| `STATUTE_DESCRIPTION_MAX_LENGTH` | int | 1000 | 법령 검색 시 description(제1조) 최대 글자 수 |
| `CASE_CONTENT_MAX_LENGTH` | int | 5000 | 판례 전문 조회 시 case_content 최대 글자 수 |

**설명:**
- `SEARCH_TOP_K`: 판례와 법령 검색 모두에 적용되는 반환 결과 개수
- `CASE_SUMMARY_MAX_LENGTH`: 판례 검색 결과의 요약문 길이 제한 (초과 시 `...` 표시)
- `STATUTE_DESCRIPTION_MAX_LENGTH`: 법령 검색 결과의 목적 조문 길이 제한 (초과 시 `...` 표시)
- `CASE_CONTENT_MAX_LENGTH`: 판례 전문 조회(get_case_content) 시 본문 길이 제한 (초과 시 `[truncated]` 표시)

---

### 5. RRF (Reciprocal Rank Fusion) 설정

| 변수명 | 타입 | 값 | 설명 |
|--------|------|-----|------|
| `RRF_K` | int | 60 | RRF 상수 (일반적으로 60) |
| `RRF_BM25_WEIGHT` | float | 1.05 | BM25 결과 가중치 |
| `RRF_VECTOR_WEIGHT` | float | 1.0 | 벡터 결과 가중치 |

**RRF 작동 원리:**
- BM25와 벡터 검색을 별도로 실행 후 순위 기반으로 융합
- 스코어 = `1/(rank + k)` 공식 사용
- BM25와 벡터의 스케일 차이 문제 해결 (BM25: 10-50, Vector: 0-1)
- 현재 가중치: BM25 = 1.05, Vector = 1.0 (BM25 약간 우선)

---

## 🔍 판례 검색 파라미터 (CASE_*)

### BM25 필드 부스팅

| 변수명 | 값 | 적용 필드 |
|--------|-----|---------|
| `CASE_NUMBER_BOOST` | 1.5 | case_number |
| `CASE_NAME_BOOST` | 0.8 | case_name |
| `CASE_REFERENCE_STATUTE_BOOST` | 3.5 | reference_statute |
| `CASE_JUDGED_STATUTE_BOOST` | 3.5 | judged_statute |
| `CASE_JUDGMENT_SUMMARY_BOOST` | 6.0 | judgment_summary |
| `CASE_CONTENT_BOOST` | 2.5 | case_content |

---

### 함수 스코어 부스팅

**⚠️ 참고:** 현재 모든 부스팅 값이 0.0으로 설정되어 있습니다 (순수 BM25+벡터만 사용)

| 변수명 | 값 | 설명 |
|--------|-----|------|
| `CASE_COURT_LEVEL_FACTOR` | 0.0 | 법원 등급 부스팅 (대법원 > 고등법원 > 지방법원) |
| `CASE_RECENCY_MAX_SCORE` | 0.0 | 최신성 최대 점수 |
| `CASE_RECENCY_DECAY` | 0.8 | 오래된 판례의 최소 비율 |
| `CASE_RECENCY_SCALE_YEARS` | 50.0 | 완전 감쇠까지 걸리는 년수 |
| `CASE_RECENCY_MISSING_DEFAULT` | 0.0 | 날짜 누락 시 기본 점수 |
| `CASE_CITATION_FACTOR` | 0.0 | 인용수 부스팅 계수 |
| `CASE_MAX_BOOST` | 0.0 | 총 부스팅 제한 |

---

### KNN (벡터 유사도 검색)

| 변수명 | 값 | 설명 |
|--------|-----|------|
| `CASE_KNN_K` | 30 | 반환할 최근접 이웃 개수 |
| `CASE_KNN_NUM_CANDIDATES` | 150 | 검색할 후보 벡터 개수 |

---

## 📜 법령 검색 파라미터 (STATUTE_*)

### BM25 필드 부스팅

| 변수명 | 값 | 적용 필드 |
|--------|-----|---------|
| `STATUTE_LAW_NAME_BOOST` | 3.0 | law_name |
| `STATUTE_ABBREVIATION_BOOST` | 2.0 | abbreviation |
| `STATUTE_DESCRIPTION_BOOST` | 1.0 | description |

---

### 함수 스코어 부스팅

| 변수명 | 값 | 설명 |
|--------|-----|------|
| `STATUTE_CITATION_FACTOR` | 0.0 | 인용수 부스팅 계수 (현재 비활성화) |
| `STATUTE_MAX_BOOST` | 0.0 | 총 부스팅 제한 |

---

### KNN (벡터 유사도 검색)

| 변수명 | 값 | 설명 |
|--------|-----|------|
| `STATUTE_KNN_K` | 30 | 반환할 최근접 이웃 개수 |
| `STATUTE_KNN_NUM_CANDIDATES` | 150 | 검색할 후보 벡터 개수 |

---

## 🎯 튜닝 가이드

### RRF 가중치 조정

```python
# BM25 키워드 검색 비중 증가
RRF_BM25_WEIGHT = 3.0  # 1.05 → 3.0

# 벡터 의미 검색 비중 증가
RRF_VECTOR_WEIGHT = 2.0  # 1.0 → 2.0
```

### 필드 부스팅 조정

```python
# 법령명 인용 판례 우선순위 상승
CASE_REFERENCE_STATUTE_BOOST = 5.0  # 3.5 → 5.0
CASE_JUDGED_STATUTE_BOOST = 5.0  # 3.5 → 5.0

# 판결 요지 중요도 증가
CASE_JUDGMENT_SUMMARY_BOOST = 8.0  # 6.0 → 8.0
```

### 검색 결과 표시 조정

```python
# 판례 요약문 길이 증가 (더 긴 요약 표시)
CASE_SUMMARY_MAX_LENGTH = 2000  # 1500 → 2000

# 법령 설명 길이 증가
STATUTE_DESCRIPTION_MAX_LENGTH = 1500  # 1000 → 1500

# 판례 전문 조회 길이 증가
CASE_CONTENT_MAX_LENGTH = 10000  # 5000 → 10000

# 검색 결과 개수 증가
SEARCH_TOP_K = 10  # 5 → 10
```

### KNN 정확도 향상

```python
# 더 정확한 벡터 검색 (성능 트레이드오프)
CASE_KNN_NUM_CANDIDATES = 300  # 150 → 300
STATUTE_KNN_NUM_CANDIDATES = 200  # 150 → 200
```

---

## ⚠️ 주의사항

1. **설정 변경 후 서버 재시작 필수**
   ```bash
   sudo systemctl restart legal-mcp
   ```

2. **RRF 가중치 권장 비율**
   - BM25 : Vector = 2:1 (현재 설정)
   - 키워드 중심: 3:1
   - 의미 중심: 1:2

3. **KNN 파라미터 제약**
   - `num_candidates >= k` 항상 유지
   - 일반적으로 `num_candidates = k × 5` 권장

---

## 🔗 관련 모듈

| 모듈 | 사용하는 설정 |
|------|-------------|
| `queries.py` | CASE_*, STATUTE_*, RRF_* |
| `rrf_fusion.py` | RRF_K, RRF_BM25_WEIGHT, RRF_VECTOR_WEIGHT |
| `search_cases.py` | SEARCH_TOP_K, RRF_* |
| `search_statutes.py` | SEARCH_TOP_K, RRF_* |
| `formatters.py` | CASE_SUMMARY_MAX_LENGTH, STATUTE_DESCRIPTION_MAX_LENGTH, CASE_CONTENT_MAX_LENGTH |

---
