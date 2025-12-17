# Core Modules Specification

Legal Search MCP 서버의 핵심 내부 모듈 명세서

---

## 📋 주요 모듈 (6개)

| 모듈 | 위치 | 역할 |
|------|------|------|
| **쿼리 빌더** | `queries.py` | RRF용 BM25/벡터 쿼리 생성 |
| **RRF 융합** | `rrf_fusion.py` | 순위 기반 결과 융합 |
| **임베딩** | `embedding.py` | Upstage API 임베딩 생성 |
| **포맷터** | `formatters.py` | 출력 포맷팅 |
| **ES 클라이언트** | `client.py` | Elasticsearch 연결 래퍼 |
| **설정** | `config.py` | 환경변수 및 RRF 설정 |

---

## 1. queries.py (쿼리 빌더)

**파일 위치:** `src/elasticsearch/queries.py`

### 📋 개요

RRF 하이브리드 검색을 위한 BM25/벡터 쿼리 생성

### 🔧 주요 메서드

#### 판례 검색 쿼리 (2개)

```python
build_bm25_only_case_query(query, reference_statute, court_name, date_from, date_to)
```
- BM25 multi-match 쿼리 생성
- 필드 부스팅: case_number^1.5, case_name^0.8, reference_statute^3.5, judged_statute^3.5, judgment_summary^6.0
- 함수 스코어 부스팅 (현재 모두 0.0)

```python
build_vector_only_case_query(embedding, reference_statute, court_name, date_from, date_to)
```
- KNN 벡터 검색 쿼리 생성
- k=30, num_candidates=150
- 필터: reference_statute, court_name, date_range

#### 법령 검색 쿼리 (2개)

```python
build_bm25_only_statute_query(query, law_type)
```
- BM25 multi-match 쿼리 생성
- 필드 부스팅: law_name^3.0, abbreviation^2.0, description

```python
build_vector_only_statute_query(embedding, law_type)
```
- KNN 벡터 검색 쿼리 생성
- k=30, num_candidates=150
- 필터: law_type

#### 법령 조문 쿼리 (2개)

```python
build_statute_content_query(statute_id, article_number, article_range)
```
- 법령 조문 조회 쿼리
- article_number 또는 article_range 지원

```python
build_statute_articles_list_query(statute_id)
```
- 법령 조문 목차 조회 쿼리
- _source 제한: clause_number, clause_title, reference_case_count, law_name

---

## 2. rrf_fusion.py (RRF 융합)

**파일 위치:** `src/utils/rrf_fusion.py`

### 📋 개요

BM25와 벡터 검색 결과를 순위 기반으로 융합하여 스케일 문제 해결

### 🔧 주요 메서드

```python
compute_rrf_score(rank: int, k: int = 60) -> float
```
- RRF 스코어 계산: `1 / (rank + k)`
- k: RRF 상수 (일반적으로 60)

```python
fuse_elasticsearch_hits(bm25_hits, vector_hits, k=60, bm25_weight=1.0, vector_weight=1.0)
```
- BM25와 벡터 결과 융합
- 각 문서의 RRF 스코어 계산 후 합산
- 가중치 적용 (BM25: 2.0, Vector: 1.0)
- RRF 메타데이터 추가 (bm25_rank, vector_rank, original_scores)

### 🔍 작동 원리

```
1. BM25 결과: [doc1(rank=0), doc2(rank=1), doc3(rank=2), ...]
   → doc1 RRF = 1/(0+60) * 2.0 = 0.0333
   → doc2 RRF = 1/(1+60) * 2.0 = 0.0328

2. Vector 결과: [doc2(rank=0), doc1(rank=1), doc4(rank=2), ...]
   → doc2 RRF = 1/(0+60) * 1.0 = 0.0167
   → doc1 RRF = 1/(1+60) * 1.0 = 0.0164

3. 융합:
   → doc1 총점 = 0.0333 + 0.0164 = 0.0497
   → doc2 총점 = 0.0328 + 0.0167 = 0.0495
   → 최종 순위: doc1 > doc2 > ...
```

---

## 3. embedding.py (임베딩 생성)

**파일 위치:** `src/utils/embedding.py`

### 📋 개요

Upstage Embedding API를 직접 호출하여 텍스트 임베딩 생성

### 🔧 주요 함수

```python
async def get_embedding(text: str) -> List[float]
```
- 입력: 검색 쿼리 텍스트
- 출력: 4096차원 임베딩 벡터
- API: Upstage embedding-query 모델

### ⚙️ 설정

| 항목 | 값 |
|------|-----|
| API 키 | 환경변수 `UPSTAGE_API_KEY` |
| 모델 | `embedding-query` |
| 엔드포인트 | `https://api.upstage.ai/v1` |
| 차원 | 4096 |

---

## 4. formatters.py (포맷터)

**파일 위치:** `src/utils/formatters.py`

### 📋 개요

모든 MCP 도구의 출력을 Markdown으로 포맷

### 🔧 주요 클래스

#### CaseFormatter (판례 포맷터)

```python
format_search_results(hits: List[Dict]) -> str
```
- 판례 검색 결과 목록 포맷

```python
format_case_content(case: Dict) -> str
```
- 판례 전문 상세 포맷

#### StatuteFormatter (법령 포맷터)

```python
format_search_results(hits: List[Dict]) -> str
```
- 법령 검색 결과 목록 포맷

```python
format_statute_content(statute_id, articles, metadata) -> str
```
- 법령 조문 내용 포맷

```python
format_articles_list(statute_id, articles, metadata) -> str
```
- 법령 조문 목차 포맷

### 📌 공통 포맷 특징

- 날짜: `YYYYMMDD` → `YYYY-MM-DD`
- 숫자: 천 단위 콤마 (예: `1,234`)
- 긴 텍스트: 자동 요약 (500자 초과 시 `...`)

---

## 5. client.py (ES 클라이언트)

**파일 위치:** `src/elasticsearch/client.py`

### 📋 개요

Elasticsearch 연결 및 검색 작업 래퍼

### 🔧 주요 메서드

```python
connect() -> AsyncElasticsearch
```
- ES 비동기 클라이언트 생성

```python
search(index, query, size=10, source=None) -> Dict
```
- ES 검색 실행
- RRF에서 BM25/벡터 각각 호출됨

```python
search_by_field(index, field, value, size=1) -> List[Dict]
```
- 특정 필드 값으로 문서 검색 (완전 일치)

```python
get_statute_metadata(statute_id) -> Dict | None
```
- 법령 메타데이터 조회 (총 조문 수, 약칭, 인용 횟수)

---

## 6. config.py (설정 관리)

**파일 위치:** `src/config.py`

### 📋 개요

환경 변수 및 RRF 설정 관리

### ⚙️ 주요 설정

#### Elasticsearch 연결
- `ES_HOST`, `ES_PORT`, `ES_SCHEME`, `ES_USER`, `ES_PASSWORD`

#### 인덱스명
- `INDEX_COURT_CASES`, `INDEX_STATUTES_METADATA`, `INDEX_STATUTES`

#### 임베딩
- `UPSTAGE_API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`

#### RRF 설정
- `RRF_K = 60`
- `RRF_BM25_WEIGHT = 2.0`
- `RRF_VECTOR_WEIGHT = 1.0`

#### 검색 파라미터
- BM25 필드 부스팅 (CASE_*, STATUTE_*)
- KNN 파라미터 (k=30, candidates=150)

---

## 모듈 간 의존성 (RRF 아키텍처)

```
tools/search_cases.py
  ├── embedding.get_embedding(query)
  ├── queries.build_bm25_only_case_query(...)
  ├── queries.build_vector_only_case_query(...)
  ├── client.search() × 2 (BM25, Vector)
  ├── rrf_fusion.fuse_elasticsearch_hits(...)
  └── formatters.case_formatter.format_search_results(...)

tools/search_statutes.py
  ├── embedding.get_embedding(query)
  ├── queries.build_bm25_only_statute_query(...)
  ├── queries.build_vector_only_statute_query(...)
  ├── client.search() × 2 (BM25, Vector)
  ├── rrf_fusion.fuse_elasticsearch_hits(...)
  └── formatters.statute_formatter.format_search_results(...)

tools/get_case_content.py
  ├── client.search_by_field()
  └── formatters.case_formatter.format_case_content(...)

tools/get_statute_content.py
  ├── queries.build_statute_content_query(...)
  ├── client.search()
  ├── client.get_statute_metadata()
  └── formatters.statute_formatter.format_statute_content(...)

tools/list_statute_articles.py
  ├── queries.build_statute_articles_list_query(...)
  ├── client.search()
  ├── client.get_statute_metadata()
  └── formatters.statute_formatter.format_articles_list(...)
```

---

## 주요 변경 사항 (RRF 전환)

### 제거된 코드
- ❌ `build_case_search_query()` (레거시 하이브리드 쿼리)
- ❌ `build_statute_search_query()` (레거시 하이브리드 쿼리)
- ❌ `BM25_WEIGHT`, `VECTOR_WEIGHT` (스코어 가중치)
- ❌ `USE_RRF` 플래그 (조건부 로직)

### 추가된 코드
- ✅ `rrf_fusion.py` (RRF 융합 모듈)
- ✅ `build_bm25_only_*_query()` (BM25 전용 쿼리)
- ✅ `build_vector_only_*_query()` (벡터 전용 쿼리)
- ✅ `RRF_K`, `RRF_BM25_WEIGHT`, `RRF_VECTOR_WEIGHT` (RRF 설정)

### 아키텍처 변경
- **Before:** 하나의 하이브리드 쿼리 (BM25 + KNN 동시 실행)
- **After:** BM25/벡터 별도 실행 → RRF 융합

---
