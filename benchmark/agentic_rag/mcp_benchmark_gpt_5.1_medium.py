#!/usr/bin/env python3
"""
OpenAI GPT-5.1을 활용한 MCP 벤치마크 스크립트
법률 문제 풀이를 위한 MCP 서버 연동 테스트
reasoning: medium
"""

import os
import csv
import json
import time
import re
import argparse
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

# 기본 설정
TOTAL_PROBLEMS = 150


class GPTBenchmark:
    def __init__(self, api_key: Optional[str] = None, mcp_server_url: Optional[str] = None):
        """
        GPT 벤치마크 초기화

        Args:
            api_key: OpenAI API 키 (None이면 환경변수에서 가져옴)
            mcp_server_url: MCP 서버 URL
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.mcp_server_url = mcp_server_url or "https://mcp.crow-tit.com/sse"
        self.results = []
        self.results_lock = threading.Lock()
        self.print_lock = threading.Lock()

    def load_benchmark_data(self, csv_path: str) -> List[Dict]:
        """
        벤치마크 CSV 파일 로드

        Args:
            csv_path: CSV 파일 경로

        Returns:
            벤치마크 데이터 리스트
        """
        data = []
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

    def extract_answer(self, response_text: str) -> Optional[str]:
        """
        응답에서 answer: N 형식의 답변 추출

        Args:
            response_text: GPT의 응답 텍스트

        Returns:
            추출된 답변 번호 (1-5) 또는 None
        """
        # answer: N 패턴 찾기 (대소문자 무시, ** 감싸기 포함)
        patterns = [
            r'answer:\s*\*\*([1-5])\*\*',  # answer: **3**
            r'Answer:\s*\*\*([1-5])\*\*',  # Answer: **3**
            r'ANSWER:\s*\*\*([1-5])\*\*',  # ANSWER: **3**
            r'답:\s*\*\*([1-5])\*\*',      # 답: **3**
            r'정답:\s*\*\*([1-5])\*\*',    # 정답: **3**
            r'answer:\s*([1-5])',          # answer: 3
            r'Answer:\s*([1-5])',          # Answer: 3
            r'ANSWER:\s*([1-5])',          # ANSWER: 3
            r'답:\s*([1-5])',              # 답: 3
            r'정답:\s*([1-5])',            # 정답: 3
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text)
            if match:
                return match.group(1)

        return None

    def grade_answer(self, extracted_answer: Optional[str], expected_answer: str) -> bool:
        """
        답변 채점

        Args:
            extracted_answer: 추출된 답변
            expected_answer: 정답

        Returns:
            정답 여부
        """
        if extracted_answer is None:
            return False

        return str(extracted_answer).strip() == str(expected_answer).strip()

    def run_single_query(self, prompt: str, category: str, expected_answer: str, idx: int = 0, total: int = 0, max_retries: int = 5) -> Dict:
        """
        단일 쿼리 실행

        Args:
            prompt: 질문 프롬프트
            category: 카테고리
            expected_answer: 예상 답변
            idx: 현재 인덱스
            total: 전체 개수
            max_retries: 최대 재시도 횟수

        Returns:
            실행 결과 딕셔너리
        """
        # 각 스레드에서 별도의 클라이언트 사용
        client = OpenAI(api_key=self.api_key)

        # 프롬프트에 출력 형식 지시 추가
        formatted_prompt = f"""{prompt}

**중요: 반드시 다음 형식으로 최종 답변을 제시하세요:**
answer: [1-5 중 하나의 숫자]

예시:
answer: 3"""

        for attempt in range(max_retries):
            start_time = time.time()
            try:
                # MCP 서버와 함께 응답 생성
                response = client.responses.create(
                    model="gpt-5.1",
                    input=formatted_prompt,
                    max_output_tokens=64000,
                    text={
                        "format": {
                            "type": "text"
                        },
                        "verbosity": "medium"
                    },
                    reasoning={
                        "effort": "medium",
                        "summary": "auto"
                    },
                    tools=[
                        {
                            "type": "mcp",
                            "server_label": "legal_mcp",
                            "server_url": self.mcp_server_url,
                            "allowed_tools": [
                                "search_cases",
                                "get_case_content",
                                "search_statutes",
                                "get_statute_content",
                                "list_statute_articles"
                            ],
                            "require_approval": "never"
                        }
                    ],
                    store=False
                )

                elapsed_time = time.time() - start_time

                # 응답에서 텍스트 추출
                response_text = response.output_text if hasattr(response, 'output_text') else ""

                # MCP 툴 사용 정보 추출 (output 리스트에서 mcp_call 타입 찾기)
                mcp_tool_uses = []
                if hasattr(response, 'output') and response.output:
                    for item in response.output:
                        if hasattr(item, 'type') and item.type == 'mcp_call':
                            mcp_tool_uses.append({
                                "id": item.id if hasattr(item, 'id') else None,
                                "name": item.name if hasattr(item, 'name') else None,
                                "arguments": item.arguments if hasattr(item, 'arguments') else None,
                                "server_label": item.server_label if hasattr(item, 'server_label') else None,
                                "status": item.status if hasattr(item, 'status') else None
                            })

                # 답변 추출 및 채점
                extracted_answer = self.extract_answer(response_text)
                is_correct = self.grade_answer(extracted_answer, expected_answer)

                # 재시도 조건 체크 (마지막 시도가 아닌 경우만)
                if attempt < max_retries - 1:
                    # 조건 1: 응답 없음
                    if not response_text:
                        wait_time = (2 ** attempt) * 10
                        with self.print_lock:
                            print(f"\n[{idx}/{total}] 응답 없음, {wait_time}초 후 재시도 ({attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue

                    # 조건 2: 답변 추출 실패
                    elif extracted_answer is None:
                        wait_time = (2 ** attempt) * 10
                        with self.print_lock:
                            print(f"\n[{idx}/{total}] 파싱 실패, {wait_time}초 후 재시도 ({attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue

                # 토큰 사용량 추출
                tokens_input = response.usage.input_tokens if hasattr(response, 'usage') else 0
                tokens_output = response.usage.output_tokens if hasattr(response, 'usage') else 0

                result = {
                    "idx": idx,
                    "category": category,
                    "full_prompt": formatted_prompt,
                    "expected_answer": expected_answer,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct,
                    "full_response": response_text,
                    "mcp_tools_used": len(mcp_tool_uses),
                    "mcp_tool_details": mcp_tool_uses,
                    "elapsed_time": elapsed_time,
                    "tokens_input": tokens_input,
                    "tokens_output": tokens_output,
                    "status": "success",
                    "retry_count": attempt
                }
                # 성공 시 루프 탈출
                break

            except Exception as e:
                elapsed_time = time.time() - start_time

                # 재시도 전 대기 (exponential backoff)
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 10  # 10초, 20초, 40초, 80초, 160초
                    with self.print_lock:
                        print(f"\n[{idx}/{total}] 에러 발생, {wait_time}초 후 재시도 ({attempt + 1}/{max_retries}): {str(e)[:100]}")
                    time.sleep(wait_time)
                    continue

                # 최종 실패
                result = {
                    "idx": idx,
                    "category": category,
                    "full_prompt": formatted_prompt,
                    "expected_answer": expected_answer,
                    "extracted_answer": None,
                    "is_correct": False,
                    "full_response": "",
                    "mcp_tools_used": 0,
                    "mcp_tool_details": [],
                    "elapsed_time": elapsed_time,
                    "tokens_input": 0,
                    "tokens_output": 0,
                    "status": "error",
                    "error": str(e),
                    "retry_count": attempt
                }

        # 결과 출력 (스레드 안전)
        with self.print_lock:
            print(f"\n[{idx}/{total}] 완료 - {category}")
            print(f"상태: {result['status']}")
            if result['status'] == 'success':
                print(f"정답: {result['expected_answer']} | 추출된 답: {result['extracted_answer']} | 채점: {'✓ 정답' if result['is_correct'] else '✗ 오답'}")
            else:
                print(f"에러: {result.get('error', 'Unknown error')}")
            print(f"소요 시간: {result['elapsed_time']:.2f}초")
            print(f"MCP 툴 사용: {result['mcp_tools_used']}회")
            print(f"토큰 사용: {result['tokens_input']} (입력) / {result['tokens_output']} (출력)")
            # GPT-5.1 가격: Input $1.25/MTok, Output $10.00/MTok
            query_input_cost = (result['tokens_input'] / 1_000_000) * 1.25
            query_output_cost = (result['tokens_output'] / 1_000_000) * 10.00
            query_total_cost = query_input_cost + query_output_cost
            print(f"비용: ${query_total_cost:.6f} (입력: ${query_input_cost:.6f} / 출력: ${query_output_cost:.6f})")

        return result

    def run_benchmark_batch(
        self,
        csv_path: str,
        start_idx: int = 0,
        batch_size: int = 50,
        max_workers: int = 10
    ) -> List[Dict]:
        """
        배치 단위로 벤치마크 실행 (병렬 처리)
        """
        all_data = self.load_benchmark_data(csv_path)
        total_problems = len(all_data)
        end_idx = min(start_idx + batch_size, total_problems)
        data = all_data[start_idx:end_idx]

        if not data:
            print("처리할 데이터가 없습니다.")
            return []

        print(f"\n{'='*60}")
        print(f"배치 실행: 문제 {start_idx + 1} ~ {end_idx} (총 {total_problems}개 중)")
        print(f"{'='*60}")
        print(f"병렬 워커: {max_workers}개")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, row in enumerate(data):
                idx = start_idx + i + 1
                future = executor.submit(
                    self.run_single_query,
                    prompt=row['prompt'],
                    category=row['category'],
                    expected_answer=row['answer'],
                    idx=idx,
                    total=total_problems
                )
                futures[future] = idx

            for future in as_completed(futures):
                result = future.result()
                with self.results_lock:
                    self.results.append(result)

        self.results.sort(key=lambda x: x.get('idx', 0))
        return self.results

    def save_results(self, output_path: str, metadata: Optional[Dict] = None):
        """
        결과를 JSON 파일로 저장

        Args:
            output_path: 출력 파일 경로
            metadata: 실험 메타데이터
        """
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

        # 채점 결과
        correct = sum(1 for r in self.results if r.get('is_correct', False))
        incorrect = sum(1 for r in self.results if r['status'] == 'success' and not r.get('is_correct', False))
        no_answer = sum(1 for r in self.results if r['status'] == 'success' and r.get('extracted_answer') is None)

        total_time = sum(r['elapsed_time'] for r in self.results)
        avg_time = total_time / total if total > 0 else 0

        total_input_tokens = sum(r['tokens_input'] for r in self.results)
        total_output_tokens = sum(r['tokens_output'] for r in self.results)

        total_mcp_uses = sum(r['mcp_tools_used'] for r in self.results)

        # 비용 계산 (GPT-5.1 가격)
        # Input: $1.25/MTok, Output: $10.00/MTok
        input_cost = (total_input_tokens / 1_000_000) * 1.25
        output_cost = (total_output_tokens / 1_000_000) * 10.00
        total_cost = input_cost + output_cost

        print("\n" + "="*60)
        print("벤치마크 결과 요약 (GPT-5.1, reasoning: medium)")
        print("="*60)
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
        print(f"입력 비용: ${input_cost:.4f}")
        print(f"출력 비용: ${output_cost:.4f}")
        print(f"총 비용: ${total_cost:.4f}")

        print(f"\n[MCP 사용]")
        print(f"MCP 툴 사용 횟수: {total_mcp_uses}")
        print(f"쿼리당 평균 MCP 툴 사용: {total_mcp_uses/total:.2f}")
        print("="*60)


def main(limit: int = None, workers: int = 3):
    """메인 실행 함수"""

    MODEL_NAME = "gpt-5.1"
    TEMPERATURE = None
    MAX_TOKENS = 64000
    REASONING_MODE = "medium"
    MAX_WORKERS = workers
    total_problems = limit if limit else TOTAL_PROBLEMS

    mcp_server_url = "https://mcp.crow-tit.com/sse"
    mcp_tools = ["search_cases", "get_case_content", "search_statutes", "get_statute_content", "list_statute_articles"]
    api_key = os.environ.get("OPENAI_API_KEY") or "your-openai-api-key"

    print(f"\n{'='*60}")
    print(f"벤치마크 시작")
    print(f"처리할 문제 수: {total_problems}")
    print(f"{'='*60}")

    start_time = datetime.now()
    metadata = {
        "experiment_name": "MCP RAG Benchmark",
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "reasoning_mode": REASONING_MODE,
        "max_workers": MAX_WORKERS,
        "mcp_server_url": mcp_server_url,
        "mcp_tools": mcp_tools,
        "start_time": start_time.isoformat(),
        "total_problems": total_problems
    }

    # 벤치마크 실행 - 항상 0부터 시작
    benchmark = GPTBenchmark(api_key=api_key, mcp_server_url=mcp_server_url)
    results = benchmark.run_benchmark_batch(
        "benchmark.csv",
        start_idx=0,
        batch_size=total_problems,
        max_workers=MAX_WORKERS
    )

    metadata["end_time"] = datetime.now().isoformat()
    metadata["elapsed_seconds"] = (datetime.now() - start_time).total_seconds()

    # 결과 저장
    benchmark.results = results
    os.makedirs("result", exist_ok=True)
    output_file = "result/mcp_benchmark_gpt_5.1_medium_result.json"
    benchmark.save_results(output_file, metadata=metadata)

    print("벤치마크 완료!")
    benchmark.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP RAG Benchmark - GPT-5.1 Medium")
    parser.add_argument("--limit", type=int, default=None, help="실행할 문제 수 (기본값: 전체)")
    parser.add_argument('--workers', type=int, default=3, help='병렬 워커 수 (기본: 3)')
    args = parser.parse_args()
    main(limit=args.limit, workers=args.workers)
