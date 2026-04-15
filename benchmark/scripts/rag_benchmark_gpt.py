#!/usr/bin/env python3
"""
통합 GPT Naive RAG 벤치마크 스크립트
GPT 모델을 모두 지원하며, effort 파라미터로 reasoning 강도를 통일 제어

사용 예시:
  # GPT-5.1 + high effort
  python rag_benchmark_gpt.py --model gpt-5.1 --effort high --workers 3

  # GPT-5.1 + none effort
  python rag_benchmark_gpt.py --model gpt-5.1 --effort none --workers 3
"""

import os
import csv
import json
import time
import re
import argparse
from typing import List, Dict, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dotenv import load_dotenv
from openai import OpenAI
from elasticsearch import Elasticsearch

# .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

# 접속 설정
ES_HOST = os.environ.get("ES_HOST") or "your-es-host"
ES_PORT = os.environ.get("ES_PORT") or "9200"
ES_SCHEME = os.environ.get("ES_SCHEME") or "http"
ES_USER = os.environ.get("ES_USER") or "your-es-user"
ES_PASSWORD = os.environ.get("ES_PASSWORD") or "your-es-password"
ES_URL = f"{ES_SCHEME}://{ES_HOST}:{ES_PORT}"
ES_AUTH = (ES_USER, ES_PASSWORD)
UPSTAGE_API_KEY = os.environ.get("UPSTAGE_API_KEY") or "your-upstage-api-key"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or "your-openai-api-key"

# 기본 설정
TOTAL_PROBLEMS = 150

# 모델별 가격 (input_per_mtok, output_per_mtok)
MODEL_PRICING = {
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-pro": (10.50, 84.00),
    "gpt-5.2": (0.88, 7.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
}

DEFAULT_PRICING = (1.25, 10.00)


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> tuple:
    """모델별 비용 계산"""
    ip, op = MODEL_PRICING.get(model, DEFAULT_PRICING)
    input_cost = (input_tokens / 1_000_000) * ip
    output_cost = (output_tokens / 1_000_000) * op
    return input_cost, output_cost


class RAGBenchmark:
    def __init__(self, model: str, effort: str, num_workers: int = 10):
        self.model = model
        self.effort = effort
        self.es = Elasticsearch(ES_URL, http_auth=ES_AUTH)
        self.upstage = OpenAI(api_key=UPSTAGE_API_KEY, base_url="https://api.upstage.ai/v1")
        self.gpt = OpenAI(api_key=OPENAI_API_KEY)
        self.results = []
        self.num_workers = num_workers
        self.results_lock = threading.Lock()
        self.print_lock = threading.Lock()
        self.completed_count = 0

    def get_embedding(self, text: str) -> List[float]:
        """Upstage embedding-query로 쿼리 벡터 생성"""
        response = self.upstage.embeddings.create(
            input=text,
            model="embedding-query"
        )
        return response.data[0].embedding

    def search_statutes(self, query_vector: List[float], k: int = 10) -> List[dict]:
        """법령 조문 검색"""
        result = self.es.search(
            index="test_statutes_v2",
            body={
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": k,
                    "num_candidates": 100
                },
                "_source": [
                    "law_name", "clause_number", "clause_title", "clause_content",
                    "law_type", "effective_date", "reference_case_count"
                ]
            }
        )
        return result["hits"]["hits"]

    def search_court_cases(self, query_vector: List[float], k: int = 10) -> List[dict]:
        """판례 검색"""
        result = self.es.search(
            index="test_court_cases_new",
            body={
                "knn": {
                    "field": "embedding_vector",
                    "query_vector": query_vector,
                    "k": k,
                    "num_candidates": 100
                },
                "_source": [
                    "case_number", "case_name", "court_name", "decision_date",
                    "judgment_summary", "reference_statute", "reference_case_count", "token_count"
                ]
            }
        )
        return result["hits"]["hits"]

    def format_cases_context(self, cases: List[dict]) -> str:
        """판례 검색 결과를 context 문자열로 포맷"""
        lines = []
        lines.append("Available judgments (top matches):")
        lines.append("""
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
""")
        for hit in cases:
            src = hit["_source"]
            score = hit["_score"]
            date_raw = src.get('decision_date', '')
            if date_raw and len(str(date_raw)) == 8:
                date_str = f"{str(date_raw)[:4]}-{str(date_raw)[4:6]}-{str(date_raw)[6:8]}"
            else:
                date_str = str(date_raw) if date_raw else 'N/A'

            summary = src.get('judgment_summary', 'N/A')
            if summary and len(summary) > 2000:
                summary = summary[:2000] + "..."

            lines.append("    ----------")
            lines.append(f"    - case_number: {src.get('case_number', 'N/A')}")
            lines.append(f"    - case_name: {src.get('case_name', 'N/A')}")
            lines.append(f"    - court_name: {src.get('court_name', 'N/A')}")
            lines.append(f"    - decision_date: {date_str}")
            lines.append(f"    - judgment_summary: {summary}")
            lines.append(f"    - reference_statutes: {src.get('reference_statute', 'N/A')}")
            lines.append(f"    - citation_count: {src.get('reference_case_count', 0)}")
            lines.append(f"    - relevance_score: {score:.4f}")
            token_count = src.get('token_count')
            lines.append(f"    - token_count: {token_count:,}" if token_count else "    - token_count: N/A")
        lines.append("    ----------")
        return "\n".join(lines)

    def format_statutes_context(self, statutes: List[dict]) -> str:
        """법령 검색 결과를 context 문자열로 포맷"""
        lines = []
        lines.append("Available statutes (top matches):")
        lines.append("""
Each result includes:
- law_name: The name of the law/act
- clause_number: The article/clause number
- clause_title: The title of the clause
- clause_content: The full text of the clause
- law_type: Type of law (법률, 시행령, 시행규칙, etc.)
- effective_date: The date the clause became effective
- citation_count: Number of court cases citing this clause
- relevance_score: Elasticsearch relevance score (higher = more relevant)
""")
        for hit in statutes:
            src = hit["_source"]
            score = hit["_score"]
            eff_date = src.get('effective_date', '')
            if eff_date and len(str(eff_date)) == 8:
                eff_date_str = f"{str(eff_date)[:4]}-{str(eff_date)[4:6]}-{str(eff_date)[6:8]}"
            else:
                eff_date_str = str(eff_date) if eff_date else 'N/A'

            lines.append("    ----------")
            lines.append(f"    - law_name: {src.get('law_name', 'N/A')}")
            lines.append(f"    - clause_number: {src.get('clause_number', 'N/A')}")
            lines.append(f"    - clause_title: {src.get('clause_title', 'N/A')}")
            lines.append(f"    - clause_content: {src.get('clause_content', 'N/A')}")
            lines.append(f"    - law_type: {src.get('law_type', 'N/A')}")
            lines.append(f"    - effective_date: {eff_date_str}")
            lines.append(f"    - citation_count: {src.get('reference_case_count', 0)}")
            lines.append(f"    - relevance_score: {score:.4f}")
        lines.append("    ----------")
        return "\n".join(lines)

    def ask_gpt(self, prompt: str, context: str) -> Dict:
        """GPT에게 문제 풀이 요청"""
        # 각 스레드에서 별도의 클라이언트 사용
        client = OpenAI(api_key=OPENAI_API_KEY)

        formatted_prompt = f"""{context}

---

{prompt}

**중요: 반드시 다음 형식으로 최종 답변을 제시하세요:**
answer: [1-5 중 하나의 숫자]

예시:
answer: 3"""

        create_params = {
            "model": self.model,
            "input": formatted_prompt,
            "max_output_tokens": 64000,
            "reasoning": {
                "effort": self.effort
            },
            "tools": [],
            "store": False
        }

        # effort가 none이면 temperature=0 설정
        if self.effort == "none":
            create_params["temperature"] = 0

        response = client.responses.create(**create_params)

        return {
            "output_text": response.output_text,
            "usage": response.usage,
            "formatted_prompt": formatted_prompt
        }

    def load_benchmark_data(self, csv_path: str) -> List[Dict]:
        data = []
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

    def extract_answer(self, response_text: str) -> Optional[str]:
        patterns = [
            r'answer:\s*\*\*([1-5])\*\*',
            r'Answer:\s*\*\*([1-5])\*\*',
            r'ANSWER:\s*\*\*([1-5])\*\*',
            r'답:\s*\*\*([1-5])\*\*',
            r'정답:\s*\*\*([1-5])\*\*',
            r'answer:\s*([1-5])',
            r'Answer:\s*([1-5])',
            r'ANSWER:\s*([1-5])',
            r'답:\s*([1-5])',
            r'정답:\s*([1-5])',
            r'정답은?\s*[①②③④⑤]\s*\(?([1-5])\)?',
            r'[①②③④⑤].*?([1-5])번',
        ]
        for pattern in patterns:
            match = re.search(pattern, response_text)
            if match:
                return match.group(1)

        # ①②③④⑤ 기호로 답변한 경우
        symbol_map = {'①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5'}
        for symbol, num in symbol_map.items():
            if f"정답: {symbol}" in response_text or f"정답은 {symbol}" in response_text:
                return num

        return None

    def grade_answer(self, extracted_answer: Optional[str], expected_answer: str) -> bool:
        if extracted_answer is None:
            return False
        return str(extracted_answer).strip() == str(expected_answer).strip()

    def run_single_query(self, prompt: str, category: str, expected_answer: str, idx: int = 0, total: int = 0, max_retries: int = 5) -> Dict:
        for attempt in range(max_retries):
            start_time = time.time()
            try:
                # 1. 임베딩 생성
                embed_start = time.time()
                query_vector = self.get_embedding(prompt)
                embed_time = time.time() - embed_start

                # 2. 검색 수행
                search_start = time.time()
                cases = self.search_court_cases(query_vector, k=10)
                statutes = self.search_statutes(query_vector, k=10)
                search_time = time.time() - search_start

                # 검색 결과 요약
                retrieved_cases = []
                for hit in cases:
                    s = hit["_source"]
                    retrieved_cases.append({
                        "case_number": s.get("case_number", "N/A"),
                        "case_name": s.get("case_name", "N/A"),
                        "court_name": s.get("court_name", "N/A"),
                        "score": round(hit["_score"], 4)
                    })
                retrieved_statutes = []
                for hit in statutes:
                    s = hit["_source"]
                    retrieved_statutes.append({
                        "law_name": s.get("law_name", "N/A"),
                        "clause_number": s.get("clause_number", "N/A"),
                        "clause_title": s.get("clause_title", "N/A"),
                        "score": round(hit["_score"], 4)
                    })

                # 3. Context 구성
                cases_context = self.format_cases_context(cases)
                statutes_context = self.format_statutes_context(statutes)
                full_context = f"{cases_context}\n\n{statutes_context}"

                # 4. LLM 호출
                llm_start = time.time()
                gpt_response = self.ask_gpt(prompt, full_context)
                llm_time = time.time() - llm_start

                response_text = gpt_response["output_text"]
                usage = gpt_response["usage"]
                formatted_prompt = gpt_response["formatted_prompt"]

                elapsed_time = time.time() - start_time

                extracted_answer = self.extract_answer(response_text)
                is_correct = self.grade_answer(extracted_answer, expected_answer)

                # 재시도 조건 체크
                if attempt < max_retries - 1:
                    if not response_text:
                        wait_time = (2 ** attempt) * 10
                        with self.print_lock:
                            print(f"\n[{idx}/{total}] 응답 없음, {wait_time}초 후 재시도 ({attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    elif extracted_answer is None:
                        wait_time = (2 ** attempt) * 10
                        with self.print_lock:
                            print(f"\n[{idx}/{total}] 파싱 실패, {wait_time}초 후 재시도 ({attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue

                result = {
                    "idx": idx,
                    "category": category,
                    "full_prompt": formatted_prompt,
                    "expected_answer": expected_answer,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct,
                    "full_response": response_text,
                    "elapsed_time": elapsed_time,
                    "embed_time": embed_time,
                    "search_time": search_time,
                    "llm_time": llm_time,
                    "tokens_input": usage.input_tokens,
                    "tokens_output": usage.output_tokens,
                    "cases_retrieved": len(cases),
                    "statutes_retrieved": len(statutes),
                    "retrieved_cases": retrieved_cases,
                    "retrieved_statutes": retrieved_statutes,
                    "status": "success",
                    "retry_count": attempt
                }
                break

            except Exception as e:
                elapsed_time = time.time() - start_time
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 10
                    with self.print_lock:
                        print(f"\n[{idx}/{total}] 에러 발생, {wait_time}초 후 재시도 ({attempt + 1}/{max_retries}): {str(e)[:100]}")
                    time.sleep(wait_time)
                    continue

                result = {
                    "idx": idx,
                    "category": category,
                    "full_prompt": prompt,
                    "expected_answer": expected_answer,
                    "extracted_answer": None,
                    "is_correct": False,
                    "full_response": "",
                    "elapsed_time": elapsed_time,
                    "embed_time": 0,
                    "search_time": 0,
                    "llm_time": 0,
                    "tokens_input": 0,
                    "tokens_output": 0,
                    "cases_retrieved": 0,
                    "statutes_retrieved": 0,
                    "status": "error",
                    "error": str(e),
                    "retry_count": attempt
                }

        # 결과 출력
        with self.print_lock:
            self.completed_count += 1
            status_icon = '✓' if result['is_correct'] else '✗'
            print(f"\n[{self.completed_count}/{total}] #{idx} {result['category'][:6]} | "
                  f"정답:{result['expected_answer']} 추출:{result['extracted_answer']} {status_icon} | "
                  f"{result['elapsed_time']:.1f}s")
            if result['status'] == 'success':
                for i, c in enumerate(result.get('retrieved_cases', []), 1):
                    print(f"  판례{i}: {c['case_number']} | {c['case_name'][:40]} (score:{c['score']})")
                for i, s in enumerate(result.get('retrieved_statutes', []), 1):
                    print(f"  법령{i}: {s['law_name']} {s['clause_number']} {s['clause_title'][:30]} (score:{s['score']})")

        return result

    def run_benchmark_batch(self, csv_path: str, start_idx: int = 0, batch_size: int = 50, save_path: str = None, metadata: Dict = None) -> List[Dict]:
        all_data = self.load_benchmark_data(csv_path)
        total_problems = len(all_data)

        # 이미 완료된 결과가 있으면 로드하여 스킵
        completed_idxs = set()
        if save_path and os.path.exists(save_path):
            try:
                with open(save_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    self.results = existing_data.get('results', [])
                    completed_idxs = {r['idx'] for r in self.results}
                    print(f"기존 결과 로드: {len(completed_idxs)}개 문제 완료됨, 나머지부터 이어서 실행")
            except (json.JSONDecodeError, KeyError):
                print("기존 결과 파일 파싱 실패, 처음부터 실행")
                self.results = []

        end_idx = min(start_idx + batch_size, total_problems)
        data = all_data[start_idx:end_idx]

        if not data:
            print("처리할 데이터가 없습니다.")
            return []

        remaining_data = [(i, row) for i, row in enumerate(data) if (start_idx + i + 1) not in completed_idxs]

        if not remaining_data:
            print("모든 문제가 이미 완료되었습니다.")
            return self.results

        self.completed_count = len(completed_idxs)

        print(f"\n{'='*60}")
        print(f"배치 실행: 문제 {start_idx + 1} ~ {end_idx} (총 {total_problems}개 중)")
        print(f"남은 문제: {len(remaining_data)}개 (완료: {len(completed_idxs)}개)")
        print(f"{'='*60}")
        print(f"병렬 워커: {self.num_workers}개")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(
                    self.run_single_query,
                    row['prompt'],
                    row['category'],
                    row['answer'],
                    start_idx + i + 1,
                    total_problems
                ): i for i, row in remaining_data
            }

            for future in as_completed(futures):
                result = future.result()
                with self.results_lock:
                    self.results.append(result)
                    if save_path:
                        self.results.sort(key=lambda x: x.get('idx', 0))
                        self.save_results(save_path, metadata)

        self.results.sort(key=lambda x: x.get('idx', 0))
        return self.results

    def save_results(self, output_path: str, metadata: Optional[Dict] = None):
        output_data = {
            "metadata": metadata or {},
            "results": self.results
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장 완료: {output_path}")

    def print_summary(self):
        if not self.results:
            print("결과가 없습니다.")
            return

        total = len(self.results)
        success = sum(1 for r in self.results if r['status'] == 'success')
        error = sum(1 for r in self.results if r['status'] == 'error')

        correct = sum(1 for r in self.results if r.get('is_correct', False))
        incorrect = sum(1 for r in self.results if r['status'] == 'success' and not r.get('is_correct', False))
        no_answer = sum(1 for r in self.results if r['status'] == 'success' and r.get('extracted_answer') is None)

        total_time = sum(r['elapsed_time'] for r in self.results)
        avg_time = total_time / total if total > 0 else 0

        total_input_tokens = sum(r['tokens_input'] for r in self.results)
        total_output_tokens = sum(r['tokens_output'] for r in self.results)

        total_cases = sum(r.get('cases_retrieved', 0) for r in self.results)
        total_statutes = sum(r.get('statutes_retrieved', 0) for r in self.results)

        total_input_cost = 0
        total_output_cost = 0
        for r in self.results:
            ic, oc = calc_cost(self.model, r['tokens_input'], r['tokens_output'])
            total_input_cost += ic
            total_output_cost += oc
        total_cost = total_input_cost + total_output_cost

        print("\n" + "=" * 60)
        print(f"Naive RAG 벤치마크 결과 요약 ({self.model}, effort: {self.effort})")
        print("=" * 60)
        print(f"총 쿼리 수: {total}")
        print(f"API 성공: {success} ({success/total*100:.1f}%)")
        print(f"API 실패: {error} ({error/total*100:.1f}%)")

        print(f"\n[채점 결과]")
        print(f"정답: {correct} ({correct/total*100:.1f}%)")
        print(f"오답: {incorrect} ({incorrect/total*100:.1f}%)")
        if no_answer > 0:
            print(f"답변 추출 실패: {no_answer} ({no_answer/total*100:.1f}%)")

        print(f"\n[성능]")
        print(f"평균 응답 시간: {avg_time:.2f}초")
        print(f"총 소요 시간: {total_time:.2f}초")

        print(f"\n[토큰 사용량]")
        print(f"총 입력 토큰: {total_input_tokens:,}")
        print(f"총 출력 토큰: {total_output_tokens:,}")
        print(f"총 토큰: {total_input_tokens + total_output_tokens:,}")

        print(f"\n[비용 추정]")
        print(f"입력 비용: ${total_input_cost:.4f}")
        print(f"출력 비용: ${total_output_cost:.4f}")
        print(f"총 비용: ${total_cost:.4f}")

        print(f"\n[검색 통계]")
        print(f"총 판례 검색: {total_cases}건")
        print(f"총 법령 검색: {total_statutes}건")
        print(f"쿼리당 평균 검색: 판례 {total_cases/total:.1f}건 / 법령 {total_statutes/total:.1f}건")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="통합 GPT Naive RAG 벤치마크")
    parser.add_argument("--model", type=str, default="gpt-5.1", help="GPT 모델명 (기본: gpt-5.1)")
    parser.add_argument("--effort", type=str, default="high", choices=["none", "low", "medium", "high"], help="reasoning 강도 (기본: high)")
    parser.add_argument("--limit", type=int, default=None, help="실행할 문제 수 (기본: 전체)")
    parser.add_argument("--workers", type=int, default=3, help="병렬 워커 수 (기본: 3)")
    parser.add_argument("--csv", type=str, default=None, help="벤치마크 CSV 경로 (기본: ../benchmark_2026.csv)")
    parser.add_argument("--output", type=str, default=None, help="결과 JSON 저장 경로 (기본: 자동 생성)")
    args = parser.parse_args()

    model = args.model
    effort = args.effort
    total_problems = args.limit if args.limit else TOTAL_PROBLEMS

    csv_path = args.csv or os.path.join(os.path.dirname(__file__), "../benchmark_2026.csv")

    if args.output:
        output_file = args.output
    else:
        result_dir = os.path.join(os.path.dirname(__file__), "../results/2026")
        model_slug = model.replace("-", "_").replace(".", "_")
        output_file = os.path.join(result_dir, f"rag_benchmark_{model_slug}_{effort}_result.json")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GPT Naive RAG 벤치마크")
    print(f"모델: {model}")
    print(f"effort: {effort}")
    print(f"처리할 문제 수: {total_problems}")
    print(f"출력 파일: {output_file}")
    print(f"{'='*60}")

    start_time = datetime.now()
    metadata = {
        "experiment_name": "Naive RAG Benchmark",
        "model": model,
        "effort": effort,
        "max_workers": args.workers,
        "elasticsearch_url": ES_URL,
        "embedding_model": "embedding-query (Upstage)",
        "start_time": start_time.isoformat(),
        "total_problems": total_problems,
    }

    benchmark = RAGBenchmark(model=model, effort=effort, num_workers=args.workers)
    results = benchmark.run_benchmark_batch(
        csv_path,
        start_idx=0,
        batch_size=total_problems,
        save_path=output_file,
        metadata=metadata
    )

    metadata["end_time"] = datetime.now().isoformat()
    metadata["elapsed_seconds"] = (datetime.now() - start_time).total_seconds()

    benchmark.results = results
    benchmark.save_results(output_file, metadata=metadata)

    print("벤치마크 완료!")
    benchmark.print_summary()


if __name__ == "__main__":
    main()
