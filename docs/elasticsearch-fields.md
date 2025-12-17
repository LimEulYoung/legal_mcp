# Elasticsearch 인덱스 필드 명세

## test_court_cases_new 인덱스

**총 193,276건** (헌법재판소 29,730건 포함)

### 필드 목록

| 필드명 | 타입 | 설명 |
|--------|------|------|
| `case_id` | keyword | 고유 ID |
| `case_number` | keyword | 사건번호 (예: 2022헌마884) |
| `case_name` | text + keyword | 사건명 (nori 분석기) |
| `case_type` | keyword | 사건 유형 (헌마, 헌바, 민사, 형사 등) |
| `court_name` | keyword | 법원명 (헌법재판소, 대법원 등) |
| `decision_date` | date (yyyyMMdd) | 선고일 |
| `court_level_score` | integer | 법원 등급 점수 (부스팅용) |
| `case_content` | text (nori) | 판결문 전문 |
| `judgment_summary` | text (nori) | 판결 요지 |
| `issues` | text (nori) | 쟁점 요약 (빈 값 많음) |
| `reference_statute` | text (nori) + keyword | 참조 법령 (keyword 서브필드 포함) |
| `judged_statute` | text (nori) + keyword | 심판대상 법령 (헌재, keyword 서브필드 포함) |
| `reference_case` | text (nori) | 참조 판례 목록 (쉼표 구분) |
| `reference_case_count` | integer | 참조 판례 개수 |
| `case_detail_link` | keyword | 법제처 URL |
| `embedding_vector` | dense_vector | Solar 4096d 임베딩 (cosine, bbq_hnsw) |
| `token_count` | integer | 토큰 개수 |

---

## test_statutes 인덱스 (조문 단위)

**총 200,633건** (법령 조문 단위)

### 필드 목록

| 필드명 | 타입 | 설명 |
|--------|------|------|
| `statute_id` | keyword | 법령 고유 ID |
| `law_name` | text + keyword | 법령명 (nori 분석기) |
| `law_type` | keyword | 법령 종류 (법률, 대통령령, 시행령 등) |
| `department` | keyword | 소관부처 |
| `abbreviation` | keyword | 약칭 (예: 개인정보보호법) |
| `revision_type_name` | keyword | 개정 유형 (전부개정, 일부개정, 타법개정 등) |
| `effective_date` | date (yyyyMMdd) | 시행일 |
| `promulgation_date` | date (yyyyMMdd) | 공포일 |
| `clause_number` | keyword | 조문 번호 (예: "1", "2의2") |
| `clause_title` | text (nori) | 조문 제목 |
| `clause_content` | text (nori) | 조문 내용 (전문) |
| `clause_count` | integer | 조문 수 |
| `reference_case_count` | integer | 참조 판례 개수 |
| `law_detail_link` | keyword | 법제처 URL |

---

## test_statutes_metadata 인덱스

**총 5,474건** (법령 단위 메타데이터)

### 필드 목록

| 필드명 | 타입 | 설명 |
|--------|------|------|
| `statute_id` | keyword | 법령 고유 ID |
| `law_name` | text + keyword | 법령명 (nori 분석기) |
| `law_type` | keyword | 법령 종류 |
| `department` | keyword | 소관부처 |
| `abbreviation` | text + keyword | 약칭 (nori 분석기) |
| `revision_type_name` | keyword | 개정 유형 |
| `effective_date` | date (yyyyMMdd) | 시행일 |
| `promulgation_date` | date (yyyyMMdd) | 공포일 |
| `clause_count` | integer | 총 조문 수 |
| `description` | text (nori) | 법령 설명 (제1조 목적 등) |
| `reference_case_count` | integer | 참조 판례 개수 |
| `law_detail_link` | keyword | 법제처 URL |
| `embedding_vector` | dense_vector | Solar 4096d 임베딩 (cosine, bbq_hnsw) |
| `token_count` | integer | 토큰 개수 |
