#!/usr/bin/env python3
"""
Ablation Experiment: get_case_content 도구 제거

이 스크립트는 agentic_rag MCP 벤치마크에서 `get_case_content` 도구를 제거했을 때
성능이 얼마나 떨어지는지 측정하는 어블레이션 실험이다.

`get_case_content`는 판례 전문을 가져오는 도구로, 이를 제거하면 모델은
`search_cases`로 검색된 판례 요약만 사용할 수 있고 전문을 깊이 조회할 수 없다.
이를 통해 판례 전문 조회(deep lookup)가 정답률에 미치는 영향을 정량화한다.

사용 예시:
  # Gemini 2.5 Pro + high effort (get_case_content 없음)
  python ablation_no_case_content_gemini.py --model gemini-2.5-pro --effort high --workers 3

  # Gemini 3 Flash + low effort
  python ablation_no_case_content_gemini.py --model gemini-3-flash-preview --effort low --workers 3
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

# effort → thinking config 매핑
# Gemini 2.5: thinking_budget (토큰 수)
EFFORT_TO_BUDGET = {
    "low": 128,
    "medium": 8192,
    "high": 32768,
    "max": 32768,
}

# Gemini 3.x: thinking_level (레벨 문자열)
EFFORT_TO_LEVEL = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "high",
}

# 모델별 가격 (input_per_mtok, output_per_mtok)
# (threshold, input_price, output_price) 형태로 구간별 가격 지원
MODEL_PRICING = {
    "gemini-2.5-pro": [
        (200000, 1.25, 10.0),   # <=200K
        (None, 2.50, 15.0),     # >200K
    ],
    "gemini-2.5-flash": [
        (200000, 0.15, 3.50),
        (None, 0.30, 7.00),
    ],
    "gemini-3-flash-preview": [
        (None, 0.25, 1.50),
    ],
    "gemini-3.1-pro-preview": [
        (200000, 2.00, 12.0),   # <=200K
        (None, 4.00, 18.0),     # >200K
    ],
    "gemini-3.1-flash-lite-preview": [
        (None, 0.25, 1.50),
    ],
}

# 기본 가격 (알 수 없는 모델)
DEFAULT_PRICING = [(None, 1.25, 10.0)]


def is_legacy_model(model: str) -> bool:
    """Gemini 2.x 모델인지 확인 (thinking_budget 사용)"""
    return "2.5" in model or "2.0" in model


def get_thinking_config(model: str, effort: str) -> types.ThinkingConfig:
    """모델과 effort에 따른 ThinkingConfig 생성"""
    if is_legacy_model(model):
        budget = EFFORT_TO_BUDGET[effort]
        return types.ThinkingConfig(
            thinking_budget=budget,
            include_thoughts=True,
        )
    else:
        level = EFFORT_TO_LEVEL[effort]
        return types.ThinkingConfig(
            thinking_level=level,
            include_thoughts=True,
        )


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> tuple:
    """모델별 비용 계산"""
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    input_cost = 0
    output_cost = 0
    for threshold, ip, op in pricing:
        if threshold is None or input_tokens <= threshold:
            input_cost = (input_tokens / 1_000_000) * ip
            output_cost = (output_tokens / 1_000_000) * op
            break
    return input_cost, output_cost


ABLATION_CONDITIONS = {
    "no_case_content": {
        "description": "판례 전문 조회(get_case_content) 제거",
        "excluded": {"get_case_content"},
    },
    "statute_only": {
        "description": "법령 도구만 제공 (판례 검색/조회 전체 제거)",
        "excluded": {"search_cases", "get_case_content"},
    },
    "case_only": {
        "description": "판례 검색만 제공 (판례 전문 조회 + 법령 도구 전체 제거)",
        "excluded": {"get_case_content", "search_statutes", "get_statute_content", "list_statute_articles"},
    },
    "full": {
        "description": "전체 도구 제공 (대조군)",
        "excluded": set(),
    },
}

# 기본값 (하위호환)
EXCLUDED_TOOLS = {"get_case_content"}


def patch_session_list_tools(session, excluded_tools):
    """세션의 list_tools를 몽키패치하여 특정 도구를 필터링"""
    original_list_tools = session.list_tools

    async def filtered_list_tools():
        result = await original_list_tools()
        result.tools = [t for t in result.tools if t.name not in excluded_tools]
        return result

    session.list_tools = filtered_list_tools


class MCPBenchmark:
    def __init__(self, model: str, effort: str):
        self.client = genai.Client()
        self.model = model
        self.effort = effort
        self.thinking_config = get_thinking_config(model, effort)
        self.results = []
        self.print_lock = asyncio.Lock()

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

    async def run_single_query(
        self,
        session,
        prompt: str,
        category: str,
        expected_answer: str,
        idx: int = 0,
        total: int = 0,
        max_retries: int = 5
    ) -> Dict:
        formatted_prompt = f"""다음 변호사 시험문제를 MCP를 활용하여 판례와 법령을 검색하여 풀이하시오.

{prompt}

**중요: 반드시 다음 형식으로 최종 답변을 제시하세요:**
answer: [1-5 중 하나의 숫자]

예시:
answer: 3"""

        for attempt in range(max_retries):
            start_time = time.time()
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=formatted_prompt,
                    config=types.GenerateContentConfig(
                        temperature=1,
                        thinking_config=self.thinking_config,
                        tools=[session],
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            maximum_remote_calls=99,
                        ),
                    ),
                )

                elapsed_time = time.time() - start_time

                response_text = response.text if response.text else ""
                thinking_text = ""
                mcp_tool_uses = []

                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if getattr(part, 'thought', False) is True and hasattr(part, 'text') and part.text:
                                    thinking_text += part.text

                if hasattr(response, 'automatic_function_calling_history') and response.automatic_function_calling_history:
                    for entry in response.automatic_function_calling_history:
                        if hasattr(entry, 'parts') and entry.parts:
                            for part in entry.parts:
                                if hasattr(part, 'function_call') and part.function_call:
                                    mcp_tool_uses.append({
                                        "name": part.function_call.name,
                                        "args": dict(part.function_call.args) if part.function_call.args else {}
                                    })

                input_tokens = 0
                output_tokens = 0
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                    output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

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

        # 결과 출력
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
            input_cost, output_cost = calc_cost(self.model, result['tokens_input'], result['tokens_output'])
            print(f"비용: ${input_cost + output_cost:.6f} (입력: ${input_cost:.6f} / 출력: ${output_cost:.6f})")

        return result

    async def run_benchmark_batch(
        self,
        csv_path: str,
        start_idx: int = 0,
        batch_size: int = 10,
        max_workers: int = 10,
        save_path: str = None,
        metadata: Dict = None,
        excluded_tools: set = None
    ) -> List[Dict]:
        if excluded_tools is None:
            excluded_tools = EXCLUDED_TOOLS
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

        print(f"\n{'='*60}")
        print(f"배치 실행: 문제 {start_idx + 1} ~ {end_idx} (총 {total_problems}개 중)")
        print(f"남은 문제: {len(remaining_data)}개 (완료: {len(completed_idxs)}개)")
        print(f"{'='*60}")
        print(f"MCP 서버 연결 중: {MCP_SERVER_URL}")

        async with sse_client(MCP_SERVER_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("MCP 세션 초기화 완료")

                patch_session_list_tools(session, excluded_tools)

                tools = await session.list_tools()
                print(f"\n사용 가능한 MCP 도구 (제외: {excluded_tools}):")
                for tool in tools.tools:
                    print(f"  - {tool.name}: {tool.description[:50] if tool.description else 'N/A'}...")

                print("\n벤치마크 실행 시작...\n")

                semaphore = asyncio.Semaphore(max_workers)
                save_lock = asyncio.Lock()

                async def run_with_semaphore(idx: int, row: Dict) -> Dict:
                    async with semaphore:
                        result = await self.run_single_query(
                            session=session,
                            prompt=row['prompt'],
                            category=row['category'],
                            expected_answer=row['answer'],
                            idx=idx,
                            total=total_problems
                        )
                        if save_path:
                            async with save_lock:
                                self.results.append(result)
                                self.results.sort(key=lambda x: x.get('idx', 0))
                                self.save_results(save_path, metadata)
                        return result

                tasks = [
                    run_with_semaphore(start_idx + i + 1, row)
                    for i, row in remaining_data
                ]
                new_results = await asyncio.gather(*tasks)

                if not save_path:
                    self.results.extend(new_results)

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


async def main():
    conditions_str = ", ".join(ABLATION_CONDITIONS.keys())
    parser = argparse.ArgumentParser(description="Ablation: Gemini MCP 벤치마크 (도구 조합별)")
    parser.add_argument("--condition", type=str, default="no_case_content",
                        choices=list(ABLATION_CONDITIONS.keys()),
                        help=f"ablation 조건: {conditions_str} (기본: no_case_content)")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="Gemini 모델명 (기본: gemini-2.5-pro)")
    parser.add_argument("--effort", type=str, default="high", choices=["low", "medium", "high", "max"], help="추론 강도 (기본: high)")
    parser.add_argument("--limit", type=int, default=None, help="실행할 문제 수 (기본: 전체)")
    parser.add_argument("--workers", type=int, default=3, help="병렬 워커 수 (기본: 3)")
    parser.add_argument("--csv", type=str, default=None, help="벤치마크 CSV 경로 (기본: ../benchmark_2026.csv)")
    parser.add_argument("--output", type=str, default=None, help="결과 JSON 저장 경로 (기본: 자동 생성)")
    args = parser.parse_args()

    condition = args.condition
    cond_info = ABLATION_CONDITIONS[condition]
    excluded_tools = cond_info["excluded"]

    model = args.model
    effort = args.effort
    total_problems = args.limit if args.limit else TOTAL_PROBLEMS

    # CSV 경로
    csv_path = args.csv or os.path.join(os.path.dirname(__file__), "../benchmark_2026.csv")

    # 연도 추출
    year = "2026"
    if csv_path and "2025" in csv_path:
        year = "2025"

    # 출력 파일 경로 자동 생성
    if args.output:
        output_file = args.output
    else:
        result_dir = os.path.join(os.path.dirname(__file__), "../results/ablation")
        model_slug = model.replace("-", "_").replace(".", "_")
        output_file = os.path.join(result_dir, f"ablation_{condition}_{model_slug}_{effort}_{year}_result.json")

    # 결과 디렉토리 생성
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # thinking config 정보 출력
    if is_legacy_model(model):
        thinking_info = f"thinking_budget={EFFORT_TO_BUDGET[effort]}"
    else:
        thinking_info = f"thinking_level={EFFORT_TO_LEVEL[effort]}"

    all_tools = ["search_cases", "get_case_content", "search_statutes", "get_statute_content", "list_statute_articles"]
    available_tools = [t for t in all_tools if t not in excluded_tools]

    print(f"\n{'='*60}")
    print(f"Ablation: {condition} ({cond_info['description']})")
    print(f"모델: {model}")
    print(f"effort: {effort} ({thinking_info})")
    print(f"처리할 문제 수: {total_problems}")
    print(f"출력 파일: {output_file}")
    print(f"제거된 도구: {excluded_tools}")
    print(f"사용 가능 도구: {available_tools}")
    print(f"{'='*60}")

    start_time = datetime.now()
    metadata = {
        "experiment_name": f"Ablation: {condition} ({cond_info['description']})",
        "experiment_type": "ablation_tool_removal",
        "condition": condition,
        "condition_description": cond_info["description"],
        "disallowed_tools": list(excluded_tools),
        "available_tools": available_tools,
        "model": model,
        "effort": effort,
        "thinking_info": thinking_info,
        "max_workers": args.workers,
        "mcp_server_url": MCP_SERVER_URL,
        "start_time": start_time.isoformat(),
        "total_problems": total_problems,
    }

    benchmark = MCPBenchmark(model=model, effort=effort)
    results = await benchmark.run_benchmark_batch(
        csv_path,
        start_idx=0,
        batch_size=total_problems,
        max_workers=args.workers,
        save_path=output_file,
        metadata=metadata,
        excluded_tools=excluded_tools
    )

    metadata["end_time"] = datetime.now().isoformat()
    metadata["elapsed_seconds"] = (datetime.now() - start_time).total_seconds()

    benchmark.results = results
    benchmark.save_results(output_file, metadata=metadata)

    print("벤치마크 완료!")
    benchmark.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
