#!/usr/bin/env python3
"""
통합 GPT MCP 벤치마크 스크립트
GPT 모델을 모두 지원하며, effort 파라미터로 reasoning 강도를 통일 제어

사용 예시:
  # GPT-5.1 + high effort
  python mcp_benchmark_gpt.py --model gpt-5.1 --effort high --workers 3

  # GPT-5.1 + none effort
  python mcp_benchmark_gpt.py --model gpt-5.1 --effort none --workers 3

  # GPT-5.1-mini + medium effort, 2025년 문제
  python mcp_benchmark_gpt.py --model gpt-5.1-mini --effort medium --csv ../benchmark_2025.csv --workers 3
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

# MCP 서버 URL
MCP_SERVER_URL = "https://mcp.crow-tit.com/sse"

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

# 기본 가격 (알 수 없는 모델)
DEFAULT_PRICING = (1.25, 10.00)


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> tuple:
    """모델별 비용 계산"""
    ip, op = MODEL_PRICING.get(model, DEFAULT_PRICING)
    input_cost = (input_tokens / 1_000_000) * ip
    output_cost = (output_tokens / 1_000_000) * op
    return input_cost, output_cost


class GPTBenchmark:
    def __init__(self, model: str, effort: str, api_key: Optional[str] = None, mcp_server_url: Optional[str] = None):
        self.model = model
        self.effort = effort
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.mcp_server_url = mcp_server_url or MCP_SERVER_URL
        self.results = []
        self.results_lock = threading.Lock()
        self.print_lock = threading.Lock()

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
        ]
        for pattern in patterns:
            match = re.search(pattern, response_text)
            if match:
                return match.group(1)
        return None

    def grade_answer(self, extracted_answer: Optional[str], expected_answer: str) -> bool:
        if extracted_answer is None:
            return False
        return str(extracted_answer).strip() == str(expected_answer).strip()

    def run_single_query(self, prompt: str, category: str, expected_answer: str, idx: int = 0, total: int = 0, max_retries: int = 5) -> Dict:
        # 각 스레드에서 별도의 클라이언트 사용
        client = OpenAI(api_key=self.api_key)

        formatted_prompt = f"""다음 변호사 시험문제를 MCP를 활용하여 판례와 법령을 검색하여 풀이하시오.

{prompt}

**중요: 반드시 다음 형식으로 최종 답변을 제시하세요:**
answer: [1-5 중 하나의 숫자]

예시:
answer: 3"""

        for attempt in range(max_retries):
            start_time = time.time()
            try:
                # reasoning effort에 따른 파라미터 구성
                create_params = {
                    "model": self.model,
                    "input": formatted_prompt,
                    "max_output_tokens": 64000,
                    "text": {
                        "format": {"type": "text"},
                        "verbosity": "medium"
                    },
                    "reasoning": {
                        "effort": self.effort,
                        "summary": "auto"
                    },
                    "tools": [
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
                    "store": False
                }

                # effort가 none이면 temperature=0 설정
                if self.effort == "none":
                    create_params["temperature"] = 0

                response = client.responses.create(**create_params)

                elapsed_time = time.time() - start_time

                # 응답에서 텍스트 추출
                response_text = response.output_text if hasattr(response, 'output_text') else ""

                # MCP 툴 사용 정보 추출
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

                # 토큰 사용량
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

        # 결과 출력
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
            input_cost, output_cost = calc_cost(self.model, result['tokens_input'], result['tokens_output'])
            print(f"비용: ${input_cost + output_cost:.6f} (입력: ${input_cost:.6f} / 출력: ${output_cost:.6f})")

        return result

    def run_benchmark_batch(
        self,
        csv_path: str,
        start_idx: int = 0,
        batch_size: int = 50,
        max_workers: int = 10,
        save_path: str = None,
        metadata: Dict = None
    ) -> List[Dict]:
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

        print(f"\n{'='*60}")
        print(f"배치 실행: 문제 {start_idx + 1} ~ {end_idx} (총 {total_problems}개 중)")
        print(f"남은 문제: {len(remaining_data)}개 (완료: {len(completed_idxs)}개)")
        print(f"{'='*60}")
        print(f"병렬 워커: {max_workers}개")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, row in remaining_data:
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

                    # 중간 저장
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

        total_mcp_uses = sum(r['mcp_tools_used'] for r in self.results)

        # 비용 계산
        total_input_cost = 0
        total_output_cost = 0
        for r in self.results:
            ic, oc = calc_cost(self.model, r['tokens_input'], r['tokens_output'])
            total_input_cost += ic
            total_output_cost += oc
        total_cost = total_input_cost + total_output_cost

        print("\n" + "="*60)
        print(f"벤치마크 결과 요약 ({self.model}, effort: {self.effort})")
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
        print(f"입력 비용: ${total_input_cost:.4f}")
        print(f"출력 비용: ${total_output_cost:.4f}")
        print(f"총 비용: ${total_cost:.4f}")

        print(f"\n[MCP 사용]")
        print(f"MCP 툴 사용 횟수: {total_mcp_uses}")
        print(f"쿼리당 평균 MCP 툴 사용: {total_mcp_uses/total:.2f}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="통합 GPT MCP 벤치마크")
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

    # CSV 경로
    csv_path = args.csv or os.path.join(os.path.dirname(__file__), "../benchmark_2026.csv")

    # 출력 파일 경로 자동 생성
    if args.output:
        output_file = args.output
    else:
        result_dir = os.path.join(os.path.dirname(__file__), "../results/2026")
        # 모델명에서 파일명 생성 (예: gpt-5.1 → gpt_5_1)
        model_slug = model.replace("-", "_").replace(".", "_")
        output_file = os.path.join(result_dir, f"mcp_benchmark_{model_slug}_{effort}_result.json")

    # 결과 디렉토리 생성
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GPT MCP 벤치마크")
    print(f"모델: {model}")
    print(f"effort: {effort}")
    print(f"처리할 문제 수: {total_problems}")
    print(f"출력 파일: {output_file}")
    print(f"{'='*60}")

    start_time = datetime.now()
    metadata = {
        "experiment_name": "MCP RAG Benchmark",
        "model": model,
        "effort": effort,
        "max_workers": args.workers,
        "mcp_server_url": MCP_SERVER_URL,
        "start_time": start_time.isoformat(),
        "total_problems": total_problems,
    }

    benchmark = GPTBenchmark(model=model, effort=effort, mcp_server_url=MCP_SERVER_URL)
    results = benchmark.run_benchmark_batch(
        csv_path,
        start_idx=0,
        batch_size=total_problems,
        max_workers=args.workers,
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
