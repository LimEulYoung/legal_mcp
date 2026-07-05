#!/usr/bin/env python3
"""
통합 Gemini Closed Book 벤치마크 스크립트
Gemini 2.5 및 3.x 모델을 모두 지원하며, effort 파라미터로 추론 강도를 통일 제어
외부 도구/검색 없이 모델 자체 지식만으로 문제 풀이

사용 예시:
  python closed_book_benchmark_gemini.py --model gemini-2.5-pro --effort high --workers 3
  python closed_book_benchmark_gemini.py --model gemini-3-flash-preview --effort low --workers 3
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
from google import genai
from google.genai import types

# .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

# API 키 설정
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY") or "your-google-api-key"

# 기본 설정
TOTAL_PROBLEMS = 150

# effort → thinking config 매핑
EFFORT_TO_BUDGET = {"low": 128, "medium": 8192, "high": 32768, "max": 32768}
EFFORT_TO_LEVEL = {"low": "low", "medium": "medium", "high": "high", "max": "high"}

# 모델별 가격
MODEL_PRICING = {
    "gemini-2.5-pro": [(200000, 1.25, 10.0), (None, 2.50, 15.0)],
    "gemini-2.5-flash": [(200000, 0.15, 3.50), (None, 0.30, 7.00)],
    "gemini-3-flash-preview": [(None, 0.25, 1.50)],
    "gemini-3.1-pro-preview": [(200000, 2.00, 12.0), (None, 4.00, 18.0)],
    "gemini-3.1-flash-lite-preview": [(None, 0.25, 1.50)],
}
DEFAULT_PRICING = [(None, 1.25, 10.0)]


def is_legacy_model(model: str) -> bool:
    return "2.5" in model or "2.0" in model


def get_thinking_config(model: str, effort: str, budget_override: int = None) -> types.ThinkingConfig:
    if is_legacy_model(model):
        budget = budget_override if budget_override is not None else EFFORT_TO_BUDGET[effort]
        return types.ThinkingConfig(thinking_budget=budget, include_thoughts=True)
    else:
        return types.ThinkingConfig(thinking_level=EFFORT_TO_LEVEL[effort], include_thoughts=True)


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> tuple:
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    for threshold, ip, op in pricing:
        if threshold is None or input_tokens <= threshold:
            return (input_tokens / 1_000_000) * ip, (output_tokens / 1_000_000) * op
    return 0, 0


class ClosedBookBenchmark:
    def __init__(self, model: str, effort: str, num_workers: int = 3, thinking_budget: int = None):
        self.gemini_client = genai.Client()
        self.model = model
        self.effort = effort
        self.thinking_budget = thinking_budget
        self.thinking_config = get_thinking_config(model, effort, thinking_budget)
        self.results = []
        self.num_workers = num_workers
        self.print_lock = asyncio.Lock()
        self.completed_count = 0

    def load_benchmark_data(self, csv_path: str) -> List[Dict]:
        data = []
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

    def extract_answer(self, response_text: str) -> Optional[str]:
        patterns = [
            r'answer:\s*\*\*([1-5])\*\*', r'Answer:\s*\*\*([1-5])\*\*', r'ANSWER:\s*\*\*([1-5])\*\*',
            r'답:\s*\*\*([1-5])\*\*', r'정답:\s*\*\*([1-5])\*\*',
            r'answer:\s*([1-5])', r'Answer:\s*([1-5])', r'ANSWER:\s*([1-5])',
            r'답:\s*([1-5])', r'정답:\s*([1-5])',
            r'정답은?\s*[①②③④⑤]\s*\(?([1-5])\)?', r'[①②③④⑤].*?([1-5])번',
        ]
        for pattern in patterns:
            match = re.search(pattern, response_text)
            if match:
                return match.group(1)
        symbol_map = {'①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5'}
        for symbol, num in symbol_map.items():
            if f"정답: {symbol}" in response_text or f"정답은 {symbol}" in response_text:
                return num
        return None

    def grade_answer(self, extracted_answer: Optional[str], expected_answer: str) -> bool:
        if extracted_answer is None:
            return False
        return str(extracted_answer).strip() == str(expected_answer).strip()

    async def ask_gemini(self, prompt: str) -> Dict:
        formatted_prompt = f"""{prompt}

**중요: 반드시 다음 형식으로 최종 답변을 제시하세요:**
answer: [1-5 중 하나의 숫자]

예시:
answer: 3"""

        response = await self.gemini_client.aio.models.generate_content(
            model=self.model,
            contents=formatted_prompt,
            config=types.GenerateContentConfig(
                temperature=1,
                thinking_config=self.thinking_config,
                tools=[],
            ),
        )

        response_text = response.text if response.text else ""
        thinking_text = ""
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if getattr(part, 'thought', False) is True and hasattr(part, 'text') and part.text:
                            thinking_text += part.text

        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

        return {"output_text": response_text, "thinking_text": thinking_text,
                "input_tokens": input_tokens, "output_tokens": output_tokens}

    async def run_single_query(self, prompt: str, category: str, expected_answer: str,
                                idx: int = 0, total: int = 0, max_retries: int = 5) -> Dict:
        for attempt in range(max_retries):
            start_time = time.time()
            try:
                gemini_response = await self.ask_gemini(prompt)
                response_text = gemini_response["output_text"]
                thinking_text = gemini_response["thinking_text"]
                input_tokens = gemini_response["input_tokens"]
                output_tokens = gemini_response["output_tokens"]
                elapsed_time = time.time() - start_time

                extracted_answer = self.extract_answer(response_text)
                is_correct = self.grade_answer(extracted_answer, expected_answer)

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
                    "idx": idx, "category": category, "prompt": prompt,
                    "expected_answer": expected_answer, "extracted_answer": extracted_answer,
                    "is_correct": is_correct, "full_response": response_text,
                    "full_thinking": thinking_text, "elapsed_time": elapsed_time,
                    "tokens_input": input_tokens, "tokens_output": output_tokens,
                    "status": "success", "retry_count": attempt
                }
                break

            except Exception as e:
                elapsed_time = time.time() - start_time
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 10
                    async with self.print_lock:
                        print(f"\n[{idx}/{total}] 에러 발생, {wait_time}초 후 재시도 ({attempt + 1}/{max_retries}): {str(e)[:100]}")
                    await asyncio.sleep(wait_time)
                    continue
                result = {
                    "idx": idx, "category": category, "prompt": prompt,
                    "expected_answer": expected_answer, "extracted_answer": None,
                    "is_correct": False, "full_response": "", "full_thinking": "",
                    "elapsed_time": elapsed_time, "tokens_input": 0, "tokens_output": 0,
                    "status": "error", "error": str(e), "retry_count": attempt
                }

        async with self.print_lock:
            self.completed_count += 1
            status_icon = '✓' if result['is_correct'] else '✗'
            print(f"[{self.completed_count}/{total}] #{idx} {result['category'][:6]} | "
                  f"정답:{result['expected_answer']} 추출:{result['extracted_answer']} {status_icon} | "
                  f"{result['elapsed_time']:.1f}s")

        return result

    async def run_benchmark_batch(self, csv_path: str, start_idx: int = 0, batch_size: int = 150,
                                   save_path: str = None, metadata: Dict = None) -> List[Dict]:
        all_data = self.load_benchmark_data(csv_path)
        total_problems = len(all_data)

        completed_idxs = set()
        if save_path and os.path.exists(save_path):
            try:
                with open(save_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    self.results = list(existing_data.get('results', []))
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

        semaphore = asyncio.Semaphore(self.num_workers)
        save_lock = asyncio.Lock()

        async def run_with_semaphore(idx: int, row: Dict) -> Dict:
            async with semaphore:
                result = await self.run_single_query(
                    prompt=row['prompt'], category=row['category'],
                    expected_answer=row['answer'], idx=idx, total=total_problems
                )
                if save_path:
                    async with save_lock:
                        self.results.append(result)
                        self.results.sort(key=lambda x: x.get('idx', 0))
                        self.save_results(save_path, metadata)
                return result

        tasks = [run_with_semaphore(start_idx + i + 1, row) for i, row in remaining_data]
        new_results = await asyncio.gather(*tasks)

        if not save_path:
            self.results.extend(new_results)

        self.results.sort(key=lambda x: x.get('idx', 0))
        return self.results

    def save_results(self, output_path: str, metadata: Optional[Dict] = None):
        output_data = {"metadata": metadata or {}, "results": self.results}
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

        total_input_cost = 0
        total_output_cost = 0
        for r in self.results:
            ic, oc = calc_cost(self.model, r['tokens_input'], r['tokens_output'])
            total_input_cost += ic
            total_output_cost += oc
        total_cost = total_input_cost + total_output_cost

        print("\n" + "="*60)
        print(f"Closed Book 벤치마크 결과 요약 ({self.model}, effort: {self.effort})")
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
        print("="*60)


async def main():
    parser = argparse.ArgumentParser(description="통합 Gemini Closed Book 벤치마크")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="Gemini 모델명 (기본: gemini-2.5-pro)")
    parser.add_argument("--effort", type=str, default="high", choices=["low", "medium", "high", "max"], help="추론 강도 (기본: high)")
    parser.add_argument("--thinking-budget", type=int, default=None, help="Gemini 2.5계열 thinking_budget 직접 지정 (effort 무시, budget 스윕용. 예: 128 512 2048 8192 32768)")
    parser.add_argument("--limit", type=int, default=None, help="실행할 문제 수 (기본: 전체)")
    parser.add_argument("--workers", type=int, default=3, help="병렬 워커 수 (기본: 3)")
    parser.add_argument("--csv", type=str, default=None, help="벤치마크 CSV 경로 (기본: ../benchmark_2026.csv)")
    parser.add_argument("--output", type=str, default=None, help="결과 JSON 저장 경로 (기본: 자동 생성)")
    args = parser.parse_args()

    model = args.model
    effort = args.effort
    thinking_budget = args.thinking_budget
    total_problems = args.limit if args.limit else TOTAL_PROBLEMS
    csv_path = args.csv or os.path.join(os.path.dirname(__file__), "../benchmark_2026.csv")

    if args.output:
        output_file = args.output
    else:
        result_dir = os.path.join(os.path.dirname(__file__), "../results/2026")
        model_slug = model.replace("-", "_").replace(".", "_")
        eff_slug = f"budget{thinking_budget}" if thinking_budget is not None else effort
        output_file = os.path.join(result_dir, f"closed_book_benchmark_{model_slug}_{eff_slug}_result.json")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    if is_legacy_model(model):
        _eff_budget = thinking_budget if thinking_budget is not None else EFFORT_TO_BUDGET[effort]
        thinking_info = f"thinking_budget={_eff_budget}"
    else:
        thinking_info = f"thinking_level={EFFORT_TO_LEVEL[effort]}"

    print(f"\n{'='*60}")
    print(f"Gemini Closed Book 벤치마크")
    print(f"모델: {model}")
    print(f"effort: {effort} ({thinking_info})")
    print(f"처리할 문제 수: {total_problems}")
    print(f"출력 파일: {output_file}")
    print(f"{'='*60}")

    start_time = datetime.now()
    metadata = {
        "experiment_name": "Closed Book Benchmark",
        "model": model, "effort": effort, "thinking_budget": thinking_budget, "thinking_info": thinking_info,
        "max_workers": args.workers, "start_time": start_time.isoformat(),
        "total_problems": total_problems,
    }

    benchmark = ClosedBookBenchmark(model=model, effort=effort, num_workers=args.workers, thinking_budget=thinking_budget)
    results = await benchmark.run_benchmark_batch(
        csv_path, start_idx=0, batch_size=total_problems,
        save_path=output_file, metadata=metadata
    )

    metadata["end_time"] = datetime.now().isoformat()
    metadata["elapsed_seconds"] = (datetime.now() - start_time).total_seconds()
    benchmark.results = results
    benchmark.save_results(output_file, metadata=metadata)

    print("벤치마크 완료!")
    benchmark.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
