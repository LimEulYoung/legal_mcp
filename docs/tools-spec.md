# MCP Tools Specification

Legal Search MCP 서버의 5개 주요 도구 명세서

## 🔍 도구 1: `search_cases` (판례 검색)

**파일 위치:** `src/tools/search_cases.py`

---

### 📋 개요

> **목적:** 자연어 질의로 관련 판례를 하이브리드 검색 (BM25 + 벡터 유사도)

---

### 📥 입력 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 | 예시 |
|---------|------|------|--------|------|------|
| `query` | string | ✅ | - | 핵심 검색 키워드 (간결하게) | `"개인정보보호법"`, `"명예훼손"` |
| `reference_statute` | string | ❌ | `None` | 법령 조문 필터 (정확한 매칭) | `"민법제911조"`, `"헌법제10조"` |
| `court_name` | string | ❌ | `None` | 법원명 필터 | `"대법원"`, `"헌법재판소"` |
| `date_from` | string | ❌ | `None` | 시작일 (YYYYMMDD) | `"20200101"` |
| `date_to` | string | ❌ | `None` | 종료일 (YYYYMMDD) | `"20231231"` |

**⚠️ 주요 변경사항 (v1.0 → v1.1):**
- ✅ `query`는 이제 **핵심 키워드만** (필터 정보 분리)
- ✅ Claude가 직접 파라미터 구조화 (LLM Parser 제거)
- ✅ 항상 하이브리드 관련성 순 정렬 (sort_by 제거)
- ✅ 결과 수는 config.SEARCH_TOP_K 고정값 사용 (기본: 10)
- ✅ **v1.1 신규:** `reference_statute` 파라미터 추가 (정확한 법령 필터링)
  - Wildcard + keyword 필드로 정확한 매칭 (예: "민법제911조" 검색 시 "형법제10조" 제외)
  - reference_statute.keyword, judged_statute.keyword 필드 사용
  - 인덱스 매핑 변경: text + keyword 서브필드

---

### 📤 출력 형식

#### 출력 필드 (Markdown 문자열)

| 필드명 | 설명 | 예시 |
|--------|------|------|
| `case_number` | 사건번호 (고유 식별자) | `2021다12345` |
| `case_name` | 사건명 | `손해배상(기)` |
| `court_name` | 법원명 | `대법원` |
| `decision_date` | 판결 날짜 (YYYY-MM-DD) | `2021-05-15` |
| `judgment_summary` | 판결 요약 | `원고의 손해배상 청구...` |
| `reference_statutes` | 인용 법령 | `민법 제750조, 개인정보보호법 제39조` |
| `citation_count` | 인용 횟수 | `342` |
| `relevance_score` | 관련성 점수 (높을수록 관련) | `23.4` |
| `token_count` | 판결문 토큰 수 | `8,521` |

<details>
<summary><strong>출력 예시 (클릭하여 펼치기)</strong></summary>

```markdown
Available judgments (top matches):

Each result includes:
- case_number: The unique identifier of the case
- case_name: The title or key issue of the case
- court_name: The name of the court that delivered the judgment
- decision_date: The date the judgment was rendered (YYYY-MM-DD)
- judgment_summary: A brief summary of the judgment
- reference_statutes: The legal provisions applied in the judgment
- citation_count: Number of times this case has been cited
- relevance_score: Elasticsearch relevance score (higher = more relevant)
- token_count: Length of the full judgment text in tokens

    ----------
    - case_number: 2021다12345
    - case_name: 손해배상(기)
    - court_name: 대법원
    - decision_date: 2021-05-15
    - judgment_summary: ...
    - reference_statutes: 민법 제750조, ...
    - citation_count: 342
    - relevance_score: 23.4
    - token_count: 8,521
    ----------
```

</details>

---

### ⚙️ 내부 처리 흐름

```
1. 임베딩 생성
   ↓ embedding.get_embedding(query)
   ↓ → 4096차원 벡터 생성 (Upstage API)

2. RRF 쿼리 생성 (BM25 + 벡터 별도 실행)
   ↓ queries.build_bm25_only_case_query(...)
   ↓ → BM25 multi-match (case_number^1.5, case_name^0.8, reference_statute^3.5,
   ↓                       judged_statute^3.5, judgment_summary^6.0, case_content^2.5)
   ↓ → 부스팅: court_level, recency, citation_count (현재 모두 0.0)
   ↓
   ↓ queries.build_vector_only_case_query(...)
   ↓ → KNN semantic search (k=30, candidates=150)
   ↓ → 필터: reference_statute (wildcard), court_name, date_range

3. Elasticsearch 검색 (2회 실행)
   ↓ bm25_response = client.search(INDEX_COURT_CASES, bm25_query, fetch_size)
   ↓ vector_response = client.search(INDEX_COURT_CASES, vector_query, fetch_size)
   ↓ fetch_size = max(top_k * 3, 50)  # RRF를 위해 더 많이 가져옴

4. RRF 융합
   ↓ rrf_fusion.fuse_elasticsearch_hits(bm25_hits, vector_hits, k=60,
   ↓                                      bm25_weight=1.05, vector_weight=1.0)
   ↓ → 순위 기반 융합 (스케일 문제 해결)
   ↓ → 최종 스코어 = 1/(rank + k) * weight
   ↓ → 상위 top_k개만 반환

5. 결과 포맷팅
   ↓ formatters.case_formatter.format_search_results(hits)
   └→ Markdown 문자열 반환
```

**🔧 RRF 아키텍처:**
- ✅ **BM25/벡터 분리 실행**: 스케일 독립적 검색
- ✅ **순위 기반 융합**: 절대 점수 대신 순위로 융합
- ✅ **스케일 문제 해결**: BM25(10-50)와 벡터(0-1)의 스케일 차이 해결

---

### 🔗 사용하는 모듈

| 모듈 | 함수 | 역할 |
|------|------|------|
| `embedding.py` | `get_embedding()` | 쿼리 벡터화 (Upstage API) |
| `queries.py` | `build_bm25_only_case_query()` | BM25 전용 쿼리 생성 |
| `queries.py` | `build_vector_only_case_query()` | 벡터 전용 쿼리 생성 |
| `client.py` | `search()` | ES 검색 실행 (2회) |
| `rrf_fusion.py` | `fuse_elasticsearch_hits()` | RRF 융합 |
| `formatters.py` | `format_search_results()` | 결과 포맷팅 |

---

---

## 📄 도구 2: `get_case_content` (판례 전문 조회)

**파일 위치:** `src/tools/get_case_content.py`

---

### 📋 개요

> **목적:** 사건번호로 판례 전문(판시사항, 판결이유 등) 조회

---

### 📥 입력 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 | 예시 |
|---------|------|------|--------|------|------|
| `case_number` | string | ✅ | - | 사건번호 | `"2021다12345"` |

---

### 📤 출력 형식

#### 출력 필드 (Markdown 문자열)

| 필드명 | 설명 | 예시 |
|--------|------|------|
| `case_number` | 사건번호 | `2021다12345` |
| `case_name` | 사건명 | `손해배상(기)` |
| `court_name` | 법원명 | `대법원` |
| `decision_date` | 판결 날짜 | `2021-05-15` |
| `token_count` | 판결문 토큰 수 | `8,521` |
| `reference_statutes` | 인용 법령 | `민법 제750조, ...` |
| `judgment_text` | 전체 판결문 내용 | `[판시사항, 판결이유 등 전문]` |

<details>
<summary><strong>출력 예시 (클릭하여 펼치기)</strong></summary>

```markdown
================
CASE: 2021다12345
================

Title: 손해배상(기)
Court: 대법원
Date: 2021-05-15
Tokens: 8,521
Relevance Statutes: 민법 제750조, ...

================================
JUDGMENT TEXT
================================

[전체 판결문 내용...]
```

</details>

---

### ⚙️ 내부 처리 흐름

```
1. 사건번호로 검색
   ↓ client.search_by_field(INDEX_COURT_CASES_NEW, "case_number", case_number)
   ↓ → term 쿼리로 정확히 일치하는 문서 조회

2. 결과 확인
   ↓ 문서가 없으면 → "Case not found" 에러 반환

3. 결과 포맷팅
   ↓ formatters.case_formatter.format_case_content(case)
   └→ Markdown 문자열 반환
```

---

### 🔗 사용하는 모듈

| 모듈 | 함수 | 역할 |
|------|------|------|
| `client.py` | `search_by_field()` | 사건번호로 판례 검색 |
| `formatters.py` | `format_case_content()` | 판례 전문 포맷팅 |

---

---

## 📜 도구 3: `search_statutes` (법령 검색)

**파일 위치:** `src/tools/search_statutes.py`

---

### 📋 개요

> **목적:** 법령명 또는 내용으로 관련 법령 검색 (하이브리드 검색)

---

### 📥 입력 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 | 예시 |
|---------|------|------|--------|------|------|
| `query` | string | ✅ | - | 핵심 검색 키워드 (간결하게) | `"개인정보보호법"`, `"택배 파손"` |
| `law_type` | string | ❌ | `None` | 법령 유형 필터 | `"법률"`, `"대통령령"` |

**⚠️ 주요 변경사항 (v1.0):**
- ✅ `query`는 이제 **핵심 키워드만** (필터 정보 분리)
- ✅ `law_type` 파라미터 추가 (법령 유형 필터링)
- ✅ Claude가 직접 파라미터 구조화 (LLM Parser 제거)
- ✅ 항상 하이브리드 관련성 순 정렬 (sort_by 제거)
- ✅ 결과 수는 config.SEARCH_TOP_K 고정값 사용 (기본: 10)

---

### 📤 출력 형식

#### 출력 필드 (Markdown 문자열)

| 필드명 | 설명 | 예시 |
|--------|------|------|
| `statute_id` | 법령 ID (고유 식별자) | `001234` |
| `law_name` | 법령명 (한글) | `개인정보 보호법` |
| `abbreviation` | 법령 약칭 | `개인정보법` |
| `law_type` | 법령 유형 | `법률`, `대통령령`, `시행규칙` |
| `clause_count` | 조문 수 | `75` |
| `description` | 법령 목적 (제1조) | `이 법은 개인정보의 처리 및 보호에 관한...` |
| `citation_count` | 인용 횟수 (판례) | `1,234` |
| `relevance_score` | 관련성 점수 | `28.3` |
| `token_count` | 법령 전문 토큰 수 | `45,621` |

<details>
<summary><strong>출력 예시 (클릭하여 펼치기)</strong></summary>

```markdown
Available Statutes (top matches):

Each result includes:
- statute_id: The unique identifier
- law_name: The name of the law in Korean
- abbreviation: Legal Abbreviation Name
- law_type: The type of law (법률, 대통령령 등)
- clause_count: Number of articles/clauses in the statute
- citation_count: Number of times cited by court cases
- relevance_score: Elasticsearch relevance score
- token_count: Length of the full statute text

----------
- statute_id: 001234
- law_name: 개인정보 보호법
- abbreviation: 개인정보법
- law_type: 법률
- clause_count: 75
- description: 이 법은 개인정보의 처리 및 보호에 관한...
- citation_count: 1,234
- relevance_score: 28.3
- token_count: 45,621
----------
```

</details>

---

### ⚙️ 내부 처리 흐름

```
1. 임베딩 생성
   ↓ embedding.get_embedding(query)
   ↓ → 4096차원 벡터 생성 (Upstage API)

2. RRF 쿼리 생성 (BM25 + 벡터 별도 실행)
   ↓ queries.build_bm25_only_statute_query(...)
   ↓ → BM25 multi-match (law_name^3.0, abbreviation^2.0, description^1.0)
   ↓ → 부스팅: citation_count (현재 0.0)
   ↓
   ↓ queries.build_vector_only_statute_query(...)
   ↓ → KNN semantic search (k=30, candidates=150)
   ↓ → 필터: law_type

3. Elasticsearch 검색 (2회 실행)
   ↓ bm25_response = client.search(INDEX_STATUTES_METADATA, bm25_query, fetch_size)
   ↓ vector_response = client.search(INDEX_STATUTES_METADATA, vector_query, fetch_size)
   ↓ fetch_size = max(top_k * 3, 50)

4. RRF 융합
   ↓ rrf_fusion.fuse_elasticsearch_hits(bm25_hits, vector_hits, k=60,
   ↓                                      bm25_weight=1.05, vector_weight=1.0)
   ↓ → 순위 기반 융합
   ↓ → 상위 top_k개만 반환

5. 결과 포맷팅
   ↓ formatters.statute_formatter.format_search_results(hits)
   └→ Markdown 문자열 반환
```

**🔧 RRF 아키텍처:**
- ✅ **BM25/벡터 분리 실행**: 스케일 독립적 검색
- ✅ **순위 기반 융합**: 절대 점수 대신 순위로 융합

---

### 🔗 사용하는 모듈

| 모듈 | 함수 | 역할 |
|------|------|------|
| `embedding.py` | `get_embedding()` | 쿼리 벡터화 (Upstage API) |
| `queries.py` | `build_bm25_only_statute_query()` | BM25 전용 쿼리 생성 |
| `queries.py` | `build_vector_only_statute_query()` | 벡터 전용 쿼리 생성 |
| `client.py` | `search()` | ES 검색 실행 (2회) |
| `rrf_fusion.py` | `fuse_elasticsearch_hits()` | RRF 융합 |
| `formatters.py` | `format_search_results()` | 결과 포맷팅 |

---

---

## 📖 도구 4: `get_statute_content` (법령 조문 조회)

**파일 위치:** `src/tools/get_statute_content.py`

---

### 📋 개요

> **목적:** 법령 ID로 전체 또는 특정 조문 내용 조회

### 📥 입력 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 | 예시 |
|---------|------|------|--------|------|------|
| `statute_id` | string | ✅ | - | 법령 ID (search_statutes 결과 또는 Quick Access 테이블) | `"1706"`, `"20547"` |
| `article_number` | string | ❌ | `null` | 특정 조문 번호 | `"15"` 또는 `"750"` |
| `article_range` | string | ❌ | `null` | 조문 범위 | `"1-10"`, `"750-760"` |

**⚠️ 주의:** `article_number`와 `article_range`는 동시에 사용 불가

**Quick Access for Major Statutes**:

빠른 조회를 위해 statute_id를 제공합니다.
예: `get_statute_content(statute_id="1706", article_number="750")`

- 헌법: 1444
- 민법: 1706
- 상법: 1702
- 민사소송법: 1700
- 형법: 1692
- 형사소송법: 1671
- 행정기본법: 14041
- 행정절차법: 1362
- 행정소송법: 1363
- 헌법재판소법: 11233

---

### 📤 출력 형식

#### 출력 필드 (Markdown 문자열)

| 필드명 | 설명 | 예시 |
|--------|------|------|
| `statute_id` | 법령 ID | `001234` |
| `law_name` | 법령명 | `개인정보 보호법` |
| `abbreviation` | 법령 약칭 | `개인정보법` |
| `law_type` | 법령 유형 | `법률` |
| `effective_date` | 시행일 | `2011-03-29` |
| `promulgation_date` | 공포일 | `2011-03-29` |
| `total_clauses` | 총 조문 수 | `75` |
| `total_citation_count` | 전체 인용 횟수 | `1,234` |
| `retrieved` | 조회한 조문 | `제15조` (또는 `전체`) |
| `statute_text` | 조문 내용 | `제15조(개인정보의 수집·이용) [인용: 523회]...` |

<details>
<summary><strong>출력 예시 (클릭하여 펼치기)</strong></summary>

```markdown
================
STATUTE: 001234
================

Law Name: 개인정보 보호법
Abbreviation: 개인정보법
Law Type: 법률
Effective Date: 2011-03-29
Promulgation Date: 2011-03-29
Total Clauses: 75
Total Citation Count: 1,234 cases
Retrieved: 제15조

================================
STATUTE TEXT
================================

제15조(개인정보의 수집·이용) [인용: 523회]
① 개인정보처리자는 다음 각 호의 어느 하나에 해당하는 경우에는...
② ...

================================
```

</details>

---

### ⚙️ 내부 처리 흐름

```
1. 조문 쿼리 생성
   ↓ queries.build_statute_content_query(statute_id, article_number, article_range)
   ↓ → article_number 지정 시: 특정 조문만
   ↓ → article_range 지정 시: 조문 범위
   ↓ → 미지정 시: 전체 조문

2. 조문 검색
   ↓ client.search(INDEX_STATUTES, query, size=1000)
   ↓ → clause_number 오름차순 정렬

3. 메타데이터 조회
   ↓ client.get_statute_metadata(statute_id)
   ↓ → 총 조문 수, 약칭, 전체 인용 횟수

4. 결과 포맷팅
   ↓ formatters.statute_formatter.format_statute_content(...)
   └→ Markdown 문자열 반환
```

---

### 🔗 사용하는 모듈

| 모듈 | 함수 | 역할 |
|------|------|------|
| `queries.py` | `build_statute_content_query()` | 조문 쿼리 생성 |
| `client.py` | `search()` | 조문 검색 |
| `client.py` | `get_statute_metadata()` | 메타데이터 조회 |
| `formatters.py` | `format_statute_content()` | 조문 내용 포맷팅 |

---

---

## 📑 도구 5: `list_statute_articles` (법령 조문 목차)

**파일 위치:** `src/tools/list_statute_articles.py`

---

### 📋 개요

> **목적:** 법령의 조문 목차 (조문 번호 + 제목 + 인용 횟수) 조회

---

### 📥 입력 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 | 예시 |
|---------|------|------|--------|------|------|
| `law_name` | string | ✅ | - | 법령명 | `"개인정보 보호법"` |

---

### 📤 출력 형식

#### 출력 필드 (Markdown 문자열)

| 필드명 | 설명 | 예시 |
|--------|------|------|
| `statute_id` | 법령 ID | `001234` |
| `law_name` | 법령명 | `개인정보 보호법` |
| `abbreviation` | 법령 약칭 | `개인정보법` |
| `total_articles` | 총 조문 수 | `75` |
| `table_of_contents` | 조문 목차 리스트 | `제1조(목적) [인용: 12회]`<br>`제2조(정의) [인용: 234회]`<br>... |

<details>
<summary><strong>출력 예시 (클릭하여 펼치기)</strong></summary>

```markdown
================================================
STATUTE ARTICLES LIST: 개인정보 보호법
Abbreviation: 개인정보법
================================================

Statute ID: 001234
Total Articles: 75
Showing: All 75 articles

------------------------------------------------
TABLE OF CONTENTS
------------------------------------------------

제1조(목적) [인용: 12회]
제2조(정의) [인용: 234회]
제3조(개인정보 보호 원칙) [인용: 89회]
...
제75조(벌칙 적용에서 공무원 의제) [인용: 5회]

================================================
```

</details>

---

### ⚙️ 내부 처리 흐름

```
1. 법령명으로 statute_id 검색
   ↓ client.search_by_field(INDEX_STATUTES_METADATA, "law_name", law_name)
   ↓ → 법령 메타데이터 조회

2. 조문 목차 쿼리 생성
   ↓ queries.build_statute_articles_list_query(statute_id)
   ↓ → _source 제한: 조문 번호, 제목, 인용 횟수만

3. 조문 목차 검색
   ↓ client.search(INDEX_STATUTES_CLAUSES, query)
   ↓ → clause_number 오름차순 정렬

4. 메타데이터 조회
   ↓ client.get_statute_metadata(statute_id)
   ↓ → 총 조문 수, 약칭

5. 결과 포맷팅
   ↓ formatters.statute_formatter.format_articles_list(...)
   └→ Markdown 문자열 반환
```

---

### 🔗 사용하는 모듈

| 모듈 | 함수 | 역할 |
|------|------|------|
| `client.py` | `search_by_field()` | 법령명으로 statute_id 조회 |
| `queries.py` | `build_statute_articles_list_query()` | 목차 쿼리 생성 |
| `client.py` | `search()` | 조문 목차 검색 |
| `client.py` | `get_statute_metadata()` | 메타데이터 조회 |
| `formatters.py` | `format_articles_list()` | 목차 포맷팅 |

---

---

## 📚 공통 사항

### 🔧 핵심 모듈

| 모듈 | 위치 | 역할 |
|------|------|------|
| **포맷터** | `src/utils/formatters.py` | 모든 출력 포맷 담당 |
| **쿼리 빌더** | `src/elasticsearch/queries.py` | RRF용 BM25/벡터 쿼리 생성 |
| **임베딩** | `src/utils/embedding.py` | Upstage API 임베딩 생성 |
| **RRF 융합** | `src/utils/rrf_fusion.py` | 순위 기반 결과 융합 |
| **ES 클라이언트** | `src/elasticsearch/client.py` | Elasticsearch 연결 래퍼 |

---

### ⚠️ 에러 처리

모든 도구는 실패 시 문자열로 에러 반환:

```
Error searching cases: Connection timeout
Error: Statute not found: 존재하지않는법령
```

---
