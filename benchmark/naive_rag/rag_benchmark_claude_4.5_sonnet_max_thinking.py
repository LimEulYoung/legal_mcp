#!/usr/bin/env python3
"""
RAG 기반 Claude Sonnet 4.5 벤치마크 스크립트
법률 문제 풀이를 위한 벡터 검색 + LLM 테스트
병렬 처리 지원 (Max Thinking 모드)
"""

import os
import csv
import json
import time
import re
from typing import List, Dict, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dotenv import load_dotenv
from openai import OpenAI
from elasticsearch import Elasticsearch
import anthropic

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
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY") or "your-anthropic-api-key"

# 기본 설정
TOTAL_PROBLEMS = 150


class RAGBenchmark:
    def __init__(self, num_workers: int = 10):
        """RAG 벤치마크 초기화"""
        self.es = Elasticsearch(ES_URL, http_auth=ES_AUTH)
        self.upstage = OpenAI(api_key=UPSTAGE_API_KEY, base_url="https://api.upstage.ai/v1")
        self.anthropic_api_key = ANTHROPIC_API_KEY
        self.results = []
        self.num_workers = num_workers
        self.lock = threading.Lock()
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

    def ask_claude(self, prompt: str, context: str) -> Dict:
        """Claude Sonnet 4.5 (max thinking)에게 문제 풀이 요청"""
        formatted_prompt = f"""{context}

---

{prompt}

**중요: 반드시 다음 형식으로 최종 답변을 제시하세요:**
answer: [1-5 중 하나의 숫자]

예시:
answer: 3"""

        # 각 스레드에서 별도의 클라이언트 사용
        client = anthropic.Anthropic(api_key=self.anthropic_api_key)

        # 스트리밍으로 응답 받기 (긴 thinking 시간 지원)
        response_text = ""
        thinking_text = ""
        input_tokens = 0
        output_tokens = 0

        with client.messages.stream(
            model="claude-sonnet-4-5-20250929",
            max_tokens=64000,
            temperature=1,  # thinking 모드에서는 temperature=1 필수
            thinking={
                "type": "enabled",
                "budget_tokens": 63999  # max_tokens - 1
            },
            messages=[
                {
                    "role": "user",
                    "content": formatted_prompt
                }
            ]
        ) as stream:
            response = stream.get_final_message()

        # 응답에서 텍스트 추출
        for block in response.content:
            if block.type == "thinking":
                thinking_text += block.thinking
            elif block.type == "text":
                response_text += block.text

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return {
            "output_text": response_text,
            "thinking_text": thinking_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    def load_benchmark_data(self, csv_path: str) -> List[Dict]:
        """벤치마크 CSV 파일 로드"""
        data = []
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

    def extract_answer(self, response_text: str) -> Optional[str]:
        """응답에서 answer: N 형식의 답변 추출"""
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
        """답변 채점"""
        if extracted_answer is None:
            return False
        return str(extracted_answer).strip() == str(expected_answer).strip()

    def run_single_query(self, prompt: str, category: str, expected_answer: str, idx: int = 0, total: int = 0, max_retries: int = 5) -> Dict:
        """단일 쿼리 실행"""
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

                # 3. Context 구성
                cases_context = self.format_cases_context(cases)
                statutes_context = self.format_statutes_context(statutes)
                full_context = f"{cases_context}\n\n{statutes_context}"

                # 4. LLM 호출
                llm_start = time.time()
                claude_response = self.ask_claude(prompt, full_context)
                llm_time = time.time() - llm_start

                response_text = claude_response["output_text"]
                thinking_text = claude_response["thinking_text"]
                input_tokens = claude_response["input_tokens"]
                output_tokens = claude_response["output_tokens"]

                elapsed_time = time.time() - start_time

                # 답변 추출 및 채점
                extracted_answer = self.extract_answer(response_text)
                is_correct = self.grade_answer(extracted_answer, expected_answer)

                # 재시도 조건 체크 (마지막 시도가 아닌 경우만)
                if attempt < max_retries - 1:
                    # 조건 1: 응답 없음
                    if not response_text:
                        wait_time = (2 ** attempt) * 10
                        with self.lock:
                            print(f"\n[{idx}/{total}] 응답 없음, {wait_time}초 후 재시도 ({attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue

                    # 조건 2: 답변 추출 실패
                    elif extracted_answer is None:
                        wait_time = (2 ** attempt) * 10
                        with self.lock:
                            print(f"\n[{idx}/{total}] 파싱 실패, {wait_time}초 후 재시도 ({attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue

                result = {
                    "idx": idx,
                    "category": category,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "expected_answer": expected_answer,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct,
                    "full_response": response_text,
                    "full_thinking": thinking_text,
                    "elapsed_time": elapsed_time,
                    "embed_time": embed_time,
                    "search_time": search_time,
                    "llm_time": llm_time,
                    "tokens_input": input_tokens,
                    "tokens_output": output_tokens,
                    "cases_retrieved": len(cases),
                    "statutes_retrieved": len(statutes),
                    "status": "success",
                    "retry_count": attempt
                }
                # 성공 시 루프 탈출
                break

            except Exception as e:
                elapsed_time = time.time() - start_time

                # 재시도 전 대기 (exponential backoff)
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 10
                    with self.lock:
                        print(f"\n[{idx}/{total}] 에러 발생, {wait_time}초 후 재시도 ({attempt + 1}/{max_retries}): {str(e)[:100]}")
                    time.sleep(wait_time)
                    continue

                # 최종 실패
                result = {
                    "idx": idx,
                    "category": category,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "expected_answer": expected_answer,
                    "extracted_answer": None,
                    "is_correct": False,
                    "full_response": "",
                    "full_thinking": "",
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

        # 진행 상황 출력 (thread-safe)
        with self.lock:
            self.completed_count += 1
            status_icon = '✓' if result['is_correct'] else '✗'
            print(f"[{self.completed_count}/{total}] #{idx} {result['category'][:6]} | "
                  f"정답:{result['expected_answer']} 추출:{result['extracted_answer']} {status_icon} | "
                  f"{result['elapsed_time']:.1f}s")

        return result

    def run_benchmark_batch(self, csv_path: str, start_idx: int = 0, batch_size: int = 50) -> List[Dict]:
        """배치 단위로 벤치마크 실행 (병렬 처리)"""
        all_data = self.load_benchmark_data(csv_path)
        total_problems = len(all_data)
        end_idx = min(start_idx + batch_size, total_problems)
        data = all_data[start_idx:end_idx]

        if not data:
            print("처리할 데이터가 없습니다.")
            return []

        self.completed_count = 0

        print(f"\n{'='*60}")
        print(f"배치 실행: 문제 {start_idx + 1} ~ {end_idx} (총 {total_problems}개 중)")
        print(f"{'='*60}")
        print(f"워커: {self.num_workers}개")

        start_time = time.time()

        # 병렬 처리
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(
                    self.run_single_query,
                    row['prompt'],
                    row['category'],
                    row['answer'],
                    start_idx + i + 1,
                    total_problems
                ): i for i, row in enumerate(data)
            }

            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)

        # idx 순서대로 정렬
        self.results.sort(key=lambda x: x['idx'])

        elapsed = time.time() - start_time
        print("=" * 60)
        print(f"배치 완료: {elapsed:.1f}초 (평균 {elapsed/len(data):.1f}초/문제)")

        return self.results

    def save_results(self, output_path: str, metadata: Optional[Dict] = None):
        """결과를 JSON 파일로 저장"""
        output_data = {
            "metadata": metadata or {},
            "results": self.results
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장 완료: {output_path}")

    def print_summary(self):
        """결과 요약 출력"""
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

        total_embed_time = sum(r['embed_time'] for r in self.results)
        total_search_time = sum(r['search_time'] for r in self.results)
        total_llm_time = sum(r['llm_time'] for r in self.results)

        total_input_tokens = sum(r['tokens_input'] for r in self.results)
        total_output_tokens = sum(r['tokens_output'] for r in self.results)

        # Claude Sonnet 4.5 가격 (Input: $3/MTok, Output: $15/MTok)
        input_cost = (total_input_tokens / 1_000_000) * 3
        output_cost = (total_output_tokens / 1_000_000) * 15
        total_cost = input_cost + output_cost

        print("\n" + "=" * 60)
        print("RAG 벤치마크 결과 요약 (Claude Sonnet 4.5 Max Thinking)")
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
        print(f"  - 임베딩: {total_embed_time:.2f}초")
        print(f"  - 검색: {total_search_time:.2f}초")
        print(f"  - LLM: {total_llm_time:.2f}초")

        print(f"\n[토큰 사용량]")
        print(f"총 입력 토큰: {total_input_tokens:,}")
        print(f"총 출력 토큰: {total_output_tokens:,}")
        print(f"총 토큰: {total_input_tokens + total_output_tokens:,}")

        print(f"\n[비용 추정] (Claude Sonnet 4.5: $3/MTok input, $15/MTok output)")
        print(f"입력 비용: ${input_cost:.4f}")
        print(f"출력 비용: ${output_cost:.4f}")
        print(f"총 비용: ${total_cost:.4f}")
        print("=" * 60)


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='RAG 벤치마크')
    parser.add_argument('--limit', type=int, default=None, help='실행할 문제 수 (기본: 전체)')
    parser.add_argument('--workers', type=int, default=3, help='병렬 워커 수 (기본: 3)')
    args = parser.parse_args()

    # 실험 설정
    MODEL_NAME = "claude-sonnet-4-5-20250929"
    TEMPERATURE = 1  # thinking 모드에서는 temperature=1 필수
    MAX_TOKENS = 64000
    THINKING_MODE = True
    THINKING_BUDGET = 63999
    MAX_WORKERS = args.workers
    total_problems = args.limit if args.limit else TOTAL_PROBLEMS

    print(f"\n{'='*60}")
    print(f"벤치마크 시작")
    print(f"처리할 문제 수: {total_problems}")
    print(f"{'='*60}")

    start_time = datetime.now()
    metadata = {
        "experiment_name": "Naive RAG Benchmark",
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "thinking_mode": THINKING_MODE,
        "thinking_budget": THINKING_BUDGET,
        "max_workers": MAX_WORKERS,
        "start_time": start_time.isoformat(),
        "total_problems": total_problems
    }

    # 벤치마크 실행 - 항상 0부터 시작
    benchmark = RAGBenchmark(num_workers=MAX_WORKERS)
    results = benchmark.run_benchmark_batch(
        "benchmark.csv",
        start_idx=0,
        batch_size=total_problems
    )

    metadata["end_time"] = datetime.now().isoformat()
    metadata["elapsed_seconds"] = (datetime.now() - start_time).total_seconds()

    # 결과 저장
    benchmark.results = results
    os.makedirs("result", exist_ok=True)
    output_file = "result/rag_benchmark_claude_4.5_sonnet_max_thinking_result.json"
    benchmark.save_results(output_file, metadata=metadata)

    print("벤치마크 완료!")
    benchmark.print_summary()


if __name__ == "__main__":
    main()
