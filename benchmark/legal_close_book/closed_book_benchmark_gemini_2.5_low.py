#!/usr/bin/env python3
"""
Gemini 2.5 Pro 단독 벤치마크 스크립트 (MCP 없음)
법률 문제 풀이 - 모델 자체 지식만 활용 (Thinking Budget: 128 - LOW)
"""

import os
import csv
import json
import time
import re
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from google import genai

# .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

# API 키 설정
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY") or "your-google-api-key"

# 기본 설정
TOTAL_PROBLEMS = 150


class GeminiBenchmarkNoMCP:
    def __init__(self, num_workers: int = 3):
        """
        Gemini 벤치마크 초기화 (MCP 없음)

        Args:
            num_workers: 병렬 워커 수
        """
        self.gemini_client = genai.Client()
        self.results = []
        self.num_workers = num_workers
        self.print_lock = asyncio.Lock()
        self.completed_count = 0

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

    async def ask_gemini(self, prompt: str) -> Dict:
        """Gemini 2.5 Pro (LOW thinking)에게 문제 풀이 요청"""
        formatted_prompt = f"""{prompt}

**중요: 반드시 다음 형식으로 최종 답변을 제시하세요:**
answer: [1-5 중 하나의 숫자]

예시:
answer: 3"""

        response = await self.gemini_client.aio.models.generate_content(
            model="gemini-2.5-pro",
            contents=formatted_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=1,
                thinking_config=genai.types.ThinkingConfig(
                    thinking_budget=128,
                    include_thoughts=True,
                ),
                tools=[],  # 인터넷 검색/외부 도구 사용 비활성화
            ),
        )

        # 응답에서 텍스트 추출
        response_text = response.text if response.text else ""
        thinking_text = ""

        # candidates에서 thinking 추출
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if getattr(part, 'thought', False) is True and hasattr(part, 'text') and part.text:
                            thinking_text += part.text

        # 토큰 사용량 추출
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

        return {
            "output_text": response_text,
            "thinking_text": thinking_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    async def run_single_query(
        self,
        prompt: str,
        category: str,
        expected_answer: str,
        idx: int = 0,
        total: int = 0,
        max_retries: int = 3
    ) -> Dict:
        """
        단일 쿼리 실행 (MCP 없음)

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
        for attempt in range(max_retries):
            start_time = time.time()

            try:
                # Gemini API 호출
                gemini_response = await self.ask_gemini(prompt)

                response_text = gemini_response["output_text"]
                thinking_text = gemini_response["thinking_text"]
                input_tokens = gemini_response["input_tokens"]
                output_tokens = gemini_response["output_tokens"]

                elapsed_time = time.time() - start_time

                # 답변 추출 및 채점
                extracted_answer = self.extract_answer(response_text)
                is_correct = self.grade_answer(extracted_answer, expected_answer)

                result = {
                    "idx": idx,
                    "category": category,
                    "prompt": prompt,
                    "expected_answer": expected_answer,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct,
                    "full_response": response_text,
                    "full_thinking": thinking_text,
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
                    wait_time = (2 ** attempt) * 5  # 5초, 10초, 20초
                    async with self.print_lock:
                        print(f"\n[{idx}/{total}] 에러 발생, {wait_time}초 후 재시도 ({attempt + 1}/{max_retries}): {str(e)[:100]}")
                    await asyncio.sleep(wait_time)
                    continue

                # 최종 실패
                result = {
                    "idx": idx,
                    "category": category,
                    "prompt": prompt,
                    "expected_answer": expected_answer,
                    "extracted_answer": None,
                    "is_correct": False,
                    "full_response": "",
                    "full_thinking": "",
                    "elapsed_time": elapsed_time,
                    "tokens_input": 0,
                    "tokens_output": 0,
                    "status": "error",
                    "error": str(e),
                    "retry_count": attempt
                }

        # 진행 상황 출력 (thread-safe)
        async with self.print_lock:
            self.completed_count += 1
            status_icon = '✓' if result['is_correct'] else '✗'
            print(f"[{self.completed_count}/{total}] #{idx} {result['category'][:6]} | "
                  f"정답:{result['expected_answer']} 추출:{result['extracted_answer']} {status_icon} | "
                  f"{result['elapsed_time']:.1f}s")

        return result

    async def run_benchmark_batch(self, csv_path: str, start_idx: int = 0, batch_size: int = 50) -> List[Dict]:
        """배치 단위로 벤치마크 실행"""
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

        start_time = time.time()
        semaphore = asyncio.Semaphore(self.num_workers)

        async def run_with_semaphore(idx: int, row: Dict) -> Dict:
            async with semaphore:
                return await self.run_single_query(
                    prompt=row['prompt'],
                    category=row['category'],
                    expected_answer=row['answer'],
                    idx=idx,
                    total=total_problems
                )

        tasks = [run_with_semaphore(start_idx + i + 1, row) for i, row in enumerate(data)]
        self.results = await asyncio.gather(*tasks)
        self.results.sort(key=lambda x: x['idx'])

        elapsed = time.time() - start_time
        print(f"배치 완료: {elapsed:.1f}초")

        return self.results

    def save_results(self, output_path: str, metadata: Optional[Dict] = None):
        """
        결과를 JSON 파일로 저장

        Args:
            output_path: 출력 파일 경로
            metadata: 메타데이터 딕셔너리
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

        # 비용 계산 (Gemini 2.5 Pro 가격: <=200K $1.25/$10, >200K $2.50/$15)
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

        print("\n" + "=" * 60)
        print("Closed Book 벤치마크 결과 요약 (Gemini 2.5 Pro LOW Thinking)")
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

        print(f"\n[비용 추정] (Gemini 2.5 Pro: <=200K $1.25/$10, >200K $2.50/$15)")
        print(f"입력 비용: ${input_cost:.4f}")
        print(f"출력 비용: ${output_cost:.4f}")
        print(f"총 비용: ${total_cost:.4f}")
        print("=" * 60)


async def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='Closed Book 벤치마크 (Gemini 2.5 Pro LOW)')
    parser.add_argument('--limit', type=int, default=None, help='실행할 문제 수 (기본: 전체)')
    parser.add_argument('--workers', type=int, default=3, help='병렬 워커 수 (기본: 3)')
    args = parser.parse_args()

    MODEL_NAME = "gemini-2.5-pro"
    THINKING_BUDGET = 128
    MAX_WORKERS = args.workers
    total_problems = args.limit if args.limit else TOTAL_PROBLEMS

    print(f"\n{'='*60}")
    print(f"Closed Book 벤치마크 시작 (Gemini 2.5 Pro LOW)")
    print(f"처리할 문제 수: {total_problems}")
    print(f"{'='*60}")

    # 실험 메타데이터
    start_time = datetime.now()
    metadata = {
        "experiment_name": "Closed Book Benchmark",
        "model": MODEL_NAME,
        "thinking_budget": THINKING_BUDGET,
        "max_workers": MAX_WORKERS,
        "start_time": start_time.isoformat(),
        "total_problems": total_problems
    }

    # 벤치마크 실행
    benchmark = GeminiBenchmarkNoMCP(num_workers=MAX_WORKERS)
    results = await benchmark.run_benchmark_batch("benchmark.csv", start_idx=0, batch_size=total_problems)

    # 메타데이터 업데이트
    metadata["end_time"] = datetime.now().isoformat()
    metadata["elapsed_seconds"] = (datetime.now() - start_time).total_seconds()

    # 결과 저장
    benchmark.results = results
    os.makedirs("result", exist_ok=True)
    output_file = "result/closed_book_benchmark_gemini_2.5_low_result.json"
    benchmark.save_results(output_file, metadata=metadata)

    # 결과 요약 출력
    print(f"\n{'='*60}")
    print("벤치마크 완료!")
    print(f"{'='*60}")
    benchmark.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
