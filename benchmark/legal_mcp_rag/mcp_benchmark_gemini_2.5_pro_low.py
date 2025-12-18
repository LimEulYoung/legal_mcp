#!/usr/bin/env python3
"""
MCP를 활용한 Gemini 2.5 Pro API 벤치마크 스크립트
법률 문제 풀이를 위한 MCP 서버 연동 테스트
Thinking Level: LOW
No Guide Added - MCP 사용 가이드 없이 순수 문제만 제공
"""

import os
import csv
import json
import time
import re
import asyncio
import argparse
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client
from google import genai
from google.genai import types

# .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

# API 키 설정
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY") or "your-google-api-key"

# MCP 서버 URL (SSE 방식)
MCP_SERVER_URL = "https://mcp.crow-tit.com/sse"

# 기본 설정
TOTAL_PROBLEMS = 150


class MCPBenchmark:
    def __init__(self):
        """
        MCP 벤치마크 초기화
        """
        self.client = genai.Client()
        self.results = []
        self.print_lock = asyncio.Lock()

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
            response_text: Gemini의 응답 텍스트

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

    async def run_single_query(
        self,
        session: ClientSession,
        prompt: str,
        category: str,
        expected_answer: str,
        idx: int = 0,
        total: int = 0,
        max_retries: int = 5
    ) -> Dict:
        """
        단일 쿼리 실행

        Args:
            session: MCP 클라이언트 세션
            prompt: 질문 프롬프트
            category: 카테고리
            expected_answer: 예상 답변
            idx: 현재 인덱스
            total: 전체 개수
            max_retries: 최대 재시도 횟수

        Returns:
            실행 결과 딕셔너리
        """
        # 프롬프트에 최소한의 MCP 가이드만 제공
        formatted_prompt = f"""다음 변호사 시험문제를 MCP를 활용하여 판례와 법령을 검색하여 풀이하시오.

{prompt}

**중요: 반드시 다음 형식으로 최종 답변을 제시하세요:**
answer: [1-5 중 하나의 숫자]

예시:
answer: 3"""

        for attempt in range(max_retries):
            start_time = time.time()
            try:
                # Gemini 모델에 MCP 도구를 연결하여 요청
                response = await self.client.aio.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=formatted_prompt,
                    config=types.GenerateContentConfig(
                        temperature=1,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=128,
                            include_thoughts=True,
                        ),
                        tools=[session],  # MCP 세션을 도구로 전달
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            maximum_remote_calls=99,
                        ),
                    ),
                )

                elapsed_time = time.time() - start_time

                # 응답에서 텍스트 추출
                response_text = response.text if response.text else ""
                thinking_text = ""
                mcp_tool_uses = []

                # candidates에서 thinking 추출
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if getattr(part, 'thought', False) is True and hasattr(part, 'text') and part.text:
                                    thinking_text += part.text

                # automatic_function_calling_history에서 MCP 도구 호출 내역 추출
                if hasattr(response, 'automatic_function_calling_history') and response.automatic_function_calling_history:
                    for entry in response.automatic_function_calling_history:
                        if hasattr(entry, 'parts') and entry.parts:
                            for part in entry.parts:
                                if hasattr(part, 'function_call') and part.function_call:
                                    mcp_tool_uses.append({
                                        "name": part.function_call.name,
                                        "args": dict(part.function_call.args) if part.function_call.args else {}
                                    })

                # 토큰 사용량 추출
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                    output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

                # 답변 추출 및 채점
                extracted_answer = self.extract_answer(response_text)
                is_correct = self.grade_answer(extracted_answer, expected_answer)

                # 응답이 비어있거나 파싱 실패시 재시도 (MCP 미사용은 재시도하지 않음)
                if attempt < max_retries - 1:
                    if not response_text:
                        wait_time = (2 ** attempt) * 10
                        async with self.print_lock:
                            print(f"\n[{idx}/{total}] 응답 없음, {wait_time}초 후 재시도 ({attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    elif extracted_answer is None:
                        wait_time = (2 ** attempt) * 10
                        async with self.print_lock:
                            print(f"\n[{idx}/{total}] 파싱 실패, {wait_time}초 후 재시도 ({attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue

                result = {
                    "idx": idx,
                    "category": category,
                    "full_prompt": formatted_prompt,
                    "expected_answer": expected_answer,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct,
                    "full_response": response_text,
                    "full_thinking": thinking_text,
                    "mcp_tools_used": len(mcp_tool_uses),
                    "mcp_tool_details": mcp_tool_uses,
                    "elapsed_time": elapsed_time,
                    "tokens_input": input_tokens,
                    "tokens_output": output_tokens,
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
                    async with self.print_lock:
                        print(f"\n[{idx}/{total}] 에러 발생, {wait_time}초 후 재시도 ({attempt + 1}/{max_retries}): {str(e)[:100]}")
                    await asyncio.sleep(wait_time)
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
                    "full_thinking": "",
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
        async with self.print_lock:
            print(f"\n[{idx}/{total}] 완료 - {category}")
            print(f"상태: {result['status']}")
            if result['status'] == 'success':
                print(f"정답: {result['expected_answer']} | 추출된 답: {result['extracted_answer']} | 채점: {'✓ 정답' if result['is_correct'] else '✗ 오답'}")
            else:
                print(f"에러: {result.get('error', 'Unknown error')}")
            print(f"소요 시간: {result['elapsed_time']:.2f}초")
            print(f"MCP 툴 사용: {result['mcp_tools_used']}회")
            print(f"토큰 사용: {result['tokens_input']} (입력) / {result['tokens_output']} (출력)")
            # Gemini 2.5 Pro 가격 (<=200K: $1.25/$10, >200K: $2.50/$15)
            if result['tokens_input'] <= 200000:
                query_input_cost = (result['tokens_input'] / 1_000_000) * 1.25
                query_output_cost = (result['tokens_output'] / 1_000_000) * 10
            else:
                query_input_cost = (result['tokens_input'] / 1_000_000) * 2.50
                query_output_cost = (result['tokens_output'] / 1_000_000) * 15
            query_total_cost = query_input_cost + query_output_cost
            print(f"비용: ${query_total_cost:.6f} (입력: ${query_input_cost:.6f} / 출력: ${query_output_cost:.6f})")

        return result

    async def run_benchmark_batch(
        self,
        csv_path: str,
        start_idx: int = 0,
        batch_size: int = 10,
        max_workers: int = 10
    ) -> List[Dict]:
        """
        배치 단위로 벤치마크 실행 (병렬 처리)

        Args:
            csv_path: CSV 파일 경로
            start_idx: 시작 인덱스 (0-based)
            batch_size: 배치 크기
            max_workers: 최대 병렬 워커 수 (기본값: 10)

        Returns:
            결과 리스트
        """
        all_data = self.load_benchmark_data(csv_path)
        total_problems = len(all_data)

        # 시작 인덱스부터 batch_size만큼 슬라이싱
        end_idx = min(start_idx + batch_size, total_problems)
        data = all_data[start_idx:end_idx]

        if not data:
            print("처리할 데이터가 없습니다.")
            return []

        print(f"\n{'='*60}")
        print(f"배치 실행: 문제 {start_idx + 1} ~ {end_idx} (총 {total_problems}개 중)")
        print(f"{'='*60}")
        print(f"MCP 서버 연결 중: {MCP_SERVER_URL}")

        # MCP 세션 연결
        async with sse_client(MCP_SERVER_URL) as (read, write):
            async with ClientSession(read, write) as session:
                # 세션 초기화
                await session.initialize()
                print("MCP 세션 초기화 완료")

                # 사용 가능한 도구 목록 확인
                tools = await session.list_tools()
                print(f"\n사용 가능한 MCP 도구:")
                for tool in tools.tools:
                    print(f"  - {tool.name}: {tool.description[:50] if tool.description else 'N/A'}...")

                print("\n벤치마크 실행 시작...\n")

                # Semaphore로 동시 실행 수 제한
                semaphore = asyncio.Semaphore(max_workers)

                async def run_with_semaphore(idx: int, row: Dict) -> Dict:
                    async with semaphore:
                        return await self.run_single_query(
                            session=session,
                            prompt=row['prompt'],
                            category=row['category'],
                            expected_answer=row['answer'],
                            idx=idx,
                            total=total_problems
                        )

                # 모든 쿼리를 병렬로 실행
                tasks = [
                    run_with_semaphore(start_idx + i + 1, row)  # 1-based 인덱스
                    for i, row in enumerate(data)
                ]
                self.results = await asyncio.gather(*tasks)

        # 결과를 원래 순서대로 정렬
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

        # 비용 계산 (Gemini 2.5 Pro 가격)
        # <=200K: $1.25/$10, >200K: $2.50/$15
        input_cost = 0
        output_cost = 0
        for r in self.results:
            if r['tokens_input'] <= 200000:
                input_cost += (r['tokens_input'] / 1_000_000) * 1.25
                output_cost += (r['tokens_output'] / 1_000_000) * 10
            else:
                input_cost += (r['tokens_input'] / 1_000_000) * 2.50
                output_cost += (r['tokens_output'] / 1_000_000) * 15
        total_cost = input_cost + output_cost

        print("\n" + "="*60)
        print("벤치마크 결과 요약")
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


async def main(limit: int = None, workers: int = 5):
    """메인 실행 함수"""

    # 실험 설정
    MODEL_NAME = "gemini-2.5-pro"
    THINKING_BUDGET = 128  # LOW에 해당하는 thinking budget
    MAX_WORKERS = workers  # 병렬 워커 수
    total_problems = limit if limit else TOTAL_PROBLEMS

    # MCP 서버 설정
    mcp_server_url = MCP_SERVER_URL

    print(f"\n{'='*60}")
    print(f"벤치마크 시작")
    print(f"처리할 문제 수: {total_problems}")
    print(f"{'='*60}")

    start_time = datetime.now()
    metadata = {
        "experiment_name": "MCP RAG Benchmark (No Guide Added)",
        "model": MODEL_NAME,
        "thinking_budget": THINKING_BUDGET,
        "max_workers": MAX_WORKERS,
        "mcp_server_url": mcp_server_url,
        "start_time": start_time.isoformat(),
        "total_problems": total_problems,
        "guide_provided": False
    }

    # 벤치마크 실행 - 항상 0부터 시작
    benchmark = MCPBenchmark()
    results = await benchmark.run_benchmark_batch(
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
    output_file = "result/mcp_benchmark_gemini_2.5_pro_low_result.json"
    benchmark.save_results(output_file, metadata=metadata)

    print("벤치마크 완료!")
    benchmark.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP RAG Benchmark - Gemini 2.5 Pro Low")
    parser.add_argument("--limit", type=int, default=None, help="실행할 문제 수 (기본값: 전체)")
    parser.add_argument('--workers', type=int, default=3, help='병렬 워커 수 (기본: 3)')
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit, workers=args.workers))
