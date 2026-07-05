#!/usr/bin/env python3
"""
Gemini Naive RAG (HYBRID retrieval) 벤치마크 스크립트  [rec1 통제실험용]

rag_benchmark_hybrid_gpt.py / rag_benchmark_hybrid_claude.py와
검색/granularity 로직 100% 동일(byte-match), LLM 호출만 Gemini(thinking budget)로 교체.

설계 (rec1 교란 통제) — GPT/Claude hybrid와 동결값 동일:
  - 판례:  test_court_cases_new — BM25 필드 부스트 + KNN(embedding_vector) → RRF
  - 법령:  test_statutes_v2 (조문 단위) — 균등 BM25(clause_content/clause_title/law_name) + KNN(embedding) → RRF
  - RRF:   k=60, bm25_weight=1.05, vector_weight=1.0
  - top_k: 10
  - granularity: summary -> judgment_summary(2000자) / full -> case_content 전문(5000자, 상위 5건)

Gemini 추론강도: thinking_budget 직접 지정(--thinking-budget). budget 사다리 = 128/512/2048/8192/32768.
환경: google-genai 1.67.0 (agentic/dense 실험과 동일 venv). ES는 환경변수(ES_HOST/ES_PASSWORD)로 주입 가능.

사용 예시:
  python rag_benchmark_hybrid_gemini.py --model gemini-2.5-pro --thinking-budget 512 --granularity full --workers 3
  python rag_benchmark_hybrid_gemini.py --model gemini-2.5-pro --thinking-budget 8192 --granularity summary --limit 10
"""

import os
import csv
import json
import time
import re
import argparse
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dotenv import load_dotenv
from openai import OpenAI              # Upstage 임베딩용
from elasticsearch import Elasticsearch
from google import genai
from google.genai import types

# .env 파일 로드 (repo root)
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

# 접속 설정 (환경변수 우선 → ES_HOST/ES_PASSWORD override 가능)
ES_HOST = os.environ.get("ES_HOST") or "your-es-host"
ES_PORT = os.environ.get("ES_PORT") or "9200"
ES_SCHEME = os.environ.get("ES_SCHEME") or "http"
ES_USER = os.environ.get("ES_USER") or "elastic"
ES_PASSWORD = os.environ.get("ES_PASSWORD") or "your-es-password"
ES_URL = f"{ES_SCHEME}://{ES_HOST}:{ES_PORT}"
UPSTAGE_API_KEY = os.environ.get("UPSTAGE_API_KEY") or "your-upstage-api-key"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or "your-google-api-key"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 인덱스 (논문 동결본)
INDEX_CASES = "test_court_cases_new"
INDEX_STATUTES_V2 = "test_statutes_v2"

# RRF 파라미터 (에이전트 config와 동일)
RRF_K = 60
RRF_BM25_WEIGHT = 1.05
RRF_VECTOR_WEIGHT = 1.0

# KNN 파라미터
KNN_K = 30
KNN_NUM_CANDIDATES = 150

# 판례 BM25 필드 부스트 (에이전트 config 값 동결)
CASE_BM25_FIELDS = [
    "case_number^1.5",
    "case_name^0.8",
    "reference_statute^3.5",
    "judged_statute^3.5",
    "judgment_summary^6.0",
    "case_content^2.5",
]
# 법령 BM25 필드 (균등 = 튜닝 없음)
STATUTE_BM25_FIELDS = ["clause_content", "clause_title", "law_name"]

# 절단 길이
CASE_SUMMARY_MAX = 2000   # 기존 Naive와 동일
CASE_FULL_MAX = 5000      # 에이전트 get_case_content(CASE_CONTENT_MAX_LENGTH)와 동일
N_FULL_CASES = 5          # full 모드: 상위 N건에만 전문 추가

TOTAL_PROBLEMS = 150

# Gemini thinking_budget 매핑 (effort fallback용; --thinking-budget이 우선)
EFFORT_TO_BUDGET = {"low": 128, "medium": 8192, "high": 32768, "max": 32768}
EFFORT_TO_LEVEL = {"low": "low", "medium": "medium", "high": "high", "max": "high"}

# 모델별 가격 (구간별 input/output per Mtok)
MODEL_PRICING = {
    "gemini-2.5-pro": [(200000, 1.25, 10.0), (None, 2.50, 15.0)],
    "gemini-2.5-flash": [(200000, 0.15, 3.50), (None, 0.30, 7.00)],
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


def rrf_fuse(bm25_hits: List[Dict], vector_hits: List[Dict],
             k: int = RRF_K, bm25_weight: float = RRF_BM25_WEIGHT,
             vector_weight: float = RRF_VECTOR_WEIGHT) -> List[Dict]:
    """src/utils/rrf_fusion.py와 동일한 순위기반 RRF 융합."""
    doc = defaultdict(lambda: {"rrf": 0.0, "hit": None})
    for rank, hit in enumerate(bm25_hits):
        d = doc[hit["_id"]]
        d["rrf"] += (1.0 / (rank + k)) * bm25_weight
        d["hit"] = hit
    for rank, hit in enumerate(vector_hits):
        d = doc[hit["_id"]]
        d["rrf"] += (1.0 / (rank + k)) * vector_weight
        if d["hit"] is None:
            d["hit"] = hit
    fused = []
    for _id, data in sorted(doc.items(), key=lambda x: x[1]["rrf"], reverse=True):
        hit = dict(data["hit"])
        hit["_score"] = data["rrf"]
        fused.append(hit)
    return fused


class HybridRAGBenchmark:
    def __init__(self, model: str, effort: str, granularity: str,
                 num_workers: int = 3, top_k: int = 10, thinking_budget: int = None):
        self.model = model
        self.effort = effort
        self.granularity = granularity  # 'summary' | 'full'
        self.top_k = top_k
        self.thinking_budget = thinking_budget
        self.thinking_config = get_thinking_config(model, effort, thinking_budget)
        self.es = Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASSWORD), request_timeout=60)
        self.upstage = OpenAI(api_key=UPSTAGE_API_KEY, base_url="https://api.upstage.ai/v1")
        self.gemini_client = genai.Client()
        self.results = []
        self.num_workers = num_workers
        self.results_lock = threading.Lock()
        self.print_lock = threading.Lock()
        self.completed_count = 0

    def get_embedding(self, text: str) -> List[float]:
        resp = self.upstage.embeddings.create(input=text, model="embedding-query")
        return resp.data[0].embedding

    # ---------------- HYBRID 검색 (GPT/Claude hybrid와 동일) ----------------
    def _search_cases_hybrid(self, query_text: str, query_vector: List[float]) -> List[dict]:
        fetch = max(self.top_k * 3, 50)
        bm25_body = {
            "query": {"bool": {"must": [
                {"multi_match": {"query": query_text, "fields": CASE_BM25_FIELDS, "type": "best_fields"}}
            ]}},
            "_source": {"excludes": ["case_content", "embedding_vector"]},
        }
        bm25 = self.es.search(index=INDEX_CASES, body=bm25_body, size=fetch)["hits"]["hits"]
        knn_body = {
            "knn": {"field": "embedding_vector", "query_vector": query_vector,
                    "k": KNN_K, "num_candidates": KNN_NUM_CANDIDATES},
            "_source": {"excludes": ["case_content", "embedding_vector"]},
        }
        vec = self.es.search(index=INDEX_CASES, body=knn_body, size=fetch)["hits"]["hits"]
        return rrf_fuse(bm25, vec)[:self.top_k]

    def _search_statutes_hybrid(self, query_text: str, query_vector: List[float]) -> List[dict]:
        fetch = max(self.top_k * 3, 50)
        src = ["law_name", "clause_number", "clause_title", "clause_content",
               "law_type", "effective_date", "reference_case_count"]
        bm25_body = {
            "query": {"bool": {"must": [
                {"multi_match": {"query": query_text, "fields": STATUTE_BM25_FIELDS, "type": "best_fields"}}
            ]}},
            "_source": src,
        }
        bm25 = self.es.search(index=INDEX_STATUTES_V2, body=bm25_body, size=fetch)["hits"]["hits"]
        knn_body = {
            "knn": {"field": "embedding", "query_vector": query_vector,
                    "k": KNN_K, "num_candidates": KNN_NUM_CANDIDATES},
            "_source": src,
        }
        vec = self.es.search(index=INDEX_STATUTES_V2, body=knn_body, size=fetch)["hits"]["hits"]
        return rrf_fuse(bm25, vec)[:self.top_k]

    def _fetch_case_contents(self, ids: List[str]) -> Dict[str, str]:
        """granularity=full일 때 top-k 판례의 전문(case_content)을 mget으로 조회."""
        if not ids:
            return {}
        body = {"docs": [{"_index": INDEX_CASES, "_id": i, "_source": ["case_content"]} for i in ids]}
        resp = self.es.mget(body=body)
        out = {}
        for d in resp["docs"]:
            if d.get("found"):
                out[d["_id"]] = (d["_source"] or {}).get("case_content", "") or ""
        return out

    # ---------------- context 포맷 (GPT/Claude hybrid와 동일) ----------------
    def format_cases_context(self, cases: List[dict], full_map: Dict[str, str]) -> str:
        lines = ["Available judgments (top matches):", """
Each result includes:
- case_number, case_name, court_name, decision_date
- judgment_summary (always provided)
- judgment_text (full judgment; additionally provided for the top cases)
- reference_statutes, citation_count, relevance_score, token_count
"""]
        for rank, hit in enumerate(cases):
            src = hit["_source"]
            date_raw = src.get("decision_date", "")
            date_str = (f"{str(date_raw)[:4]}-{str(date_raw)[4:6]}-{str(date_raw)[6:8]}"
                        if date_raw and len(str(date_raw)) == 8 else (str(date_raw) or "N/A"))
            lines.append("    ----------")
            lines.append(f"    - case_number: {src.get('case_number', 'N/A')}")
            lines.append(f"    - case_name: {src.get('case_name', 'N/A')}")
            lines.append(f"    - court_name: {src.get('court_name', 'N/A')}")
            lines.append(f"    - decision_date: {date_str}")
            summary = src.get("judgment_summary", "N/A") or "N/A"
            if len(summary) > CASE_SUMMARY_MAX:
                summary = summary[:CASE_SUMMARY_MAX] + "..."
            lines.append(f"    - judgment_summary: {summary}")
            if self.granularity == "full" and rank < N_FULL_CASES:
                body = full_map.get(hit["_id"], "") or ""
                if body:
                    if len(body) > CASE_FULL_MAX:
                        body = body[:CASE_FULL_MAX] + "..."
                    lines.append(f"    - judgment_text: {body}")
            lines.append(f"    - reference_statutes: {src.get('reference_statute', 'N/A')}")
            lines.append(f"    - citation_count: {src.get('reference_case_count', 0)}")
            lines.append(f"    - relevance_score: {hit.get('_score', 0):.4f}")
            tc = src.get("token_count")
            lines.append(f"    - token_count: {tc:,}" if tc else "    - token_count: N/A")
        lines.append("    ----------")
        return "\n".join(lines)

    def format_statutes_context(self, statutes: List[dict]) -> str:
        lines = ["Available statutes (top matches):", """
Each result includes:
- law_name, clause_number, clause_title, clause_content
- law_type, effective_date, citation_count, relevance_score
"""]
        for hit in statutes:
            src = hit["_source"]
            eff = src.get("effective_date", "")
            eff_str = (f"{str(eff)[:4]}-{str(eff)[4:6]}-{str(eff)[6:8]}"
                       if eff and len(str(eff)) == 8 else (str(eff) or "N/A"))
            lines.append("    ----------")
            lines.append(f"    - law_name: {src.get('law_name', 'N/A')}")
            lines.append(f"    - clause_number: {src.get('clause_number', 'N/A')}")
            lines.append(f"    - clause_title: {src.get('clause_title', 'N/A')}")
            lines.append(f"    - clause_content: {src.get('clause_content', 'N/A')}")
            lines.append(f"    - law_type: {src.get('law_type', 'N/A')}")
            lines.append(f"    - effective_date: {eff_str}")
            lines.append(f"    - citation_count: {src.get('reference_case_count', 0)}")
            lines.append(f"    - relevance_score: {hit.get('_score', 0):.4f}")
        lines.append("    ----------")
        return "\n".join(lines)

    # ---------------- LLM 호출 (Gemini + thinking, dense/agentic gemini와 동일 방식) ----------------
    def ask_gemini(self, prompt: str, context: str) -> Dict:
        formatted_prompt = f"""{context}

---

{prompt}

**중요: 반드시 다음 형식으로 최종 답변을 제시하세요:**
answer: [1-5 중 하나의 숫자]

예시:
answer: 3"""
        response = self.gemini_client.models.generate_content(
            model=self.model,
            contents=formatted_prompt,
            config=types.GenerateContentConfig(
                temperature=1,
                thinking_config=self.thinking_config,
                max_output_tokens=64000,
            ),
        )
        text = response.text if response.text else ""
        it = ot = 0
        um = getattr(response, "usage_metadata", None)
        if um:
            it = getattr(um, "prompt_token_count", 0) or 0
            ot = getattr(um, "candidates_token_count", 0) or 0
        return {"output_text": text, "tokens_input": it, "tokens_output": ot,
                "formatted_prompt": formatted_prompt}

    def load_benchmark_data(self, csv_path: str) -> List[Dict]:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def extract_answer(self, t: str) -> Optional[str]:
        for p in [r'answer:\s*\*\*([1-5])\*\*', r'answer:\s*([1-5])', r'정답:\s*([1-5])',
                  r'Answer:\s*([1-5])', r'ANSWER:\s*([1-5])', r'답:\s*([1-5])']:
            m = re.search(p, t)
            if m:
                return m.group(1)
        sym = {'①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5'}
        for s, n in sym.items():
            if f"정답: {s}" in t or f"정답은 {s}" in t:
                return n
        return None

    def grade_answer(self, ext: Optional[str], exp: str) -> bool:
        return ext is not None and str(ext).strip() == str(exp).strip()

    def run_single_query(self, prompt: str, category: str, expected: str,
                         idx: int = 0, total: int = 0, max_retries: int = 5) -> Dict:
        result = None
        for attempt in range(max_retries):
            start = time.time()
            try:
                qv = self.get_embedding(prompt)
                s_start = time.time()
                cases = self._search_cases_hybrid(prompt, qv)
                statutes = self._search_statutes_hybrid(prompt, qv)
                full_map = self._fetch_case_contents([h["_id"] for h in cases[:N_FULL_CASES]]) if self.granularity == "full" else {}
                search_time = time.time() - s_start

                cases_ctx = self.format_cases_context(cases, full_map)
                stat_ctx = self.format_statutes_context(statutes)
                full_context = f"{cases_ctx}\n\n{stat_ctx}"

                l_start = time.time()
                gem = self.ask_gemini(prompt, full_context)
                llm_time = time.time() - l_start

                text = gem["output_text"]
                ext = self.extract_answer(text)
                ok = self.grade_answer(ext, expected)

                if attempt < max_retries - 1 and (not text or ext is None):
                    wait = (2 ** attempt) * 10
                    with self.print_lock:
                        print(f"\n[{idx}/{total}] 응답/파싱 실패, {wait}s 후 재시도 ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue

                result = {
                    "idx": idx, "category": category, "full_prompt": gem["formatted_prompt"],
                    "expected_answer": expected, "extracted_answer": ext, "is_correct": ok,
                    "full_response": text, "elapsed_time": time.time() - start,
                    "search_time": search_time, "llm_time": llm_time,
                    "tokens_input": gem["tokens_input"], "tokens_output": gem["tokens_output"],
                    "cases_retrieved": len(cases), "statutes_retrieved": len(statutes),
                    "retrieved_cases": [{"case_number": h["_source"].get("case_number"),
                                          "score": round(h.get("_score", 0), 6)} for h in cases],
                    "retrieved_statutes": [{"law_name": h["_source"].get("law_name"),
                                             "clause_number": h["_source"].get("clause_number"),
                                             "score": round(h.get("_score", 0), 6)} for h in statutes],
                    "status": "success", "retry_count": attempt,
                }
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = (2 ** attempt) * 10
                    with self.print_lock:
                        print(f"\n[{idx}/{total}] 에러, {wait}s 후 재시도 ({attempt+1}/{max_retries}): {str(e)[:120]}")
                    time.sleep(wait)
                    continue
                result = {"idx": idx, "category": category, "full_prompt": prompt,
                          "expected_answer": expected, "extracted_answer": None, "is_correct": False,
                          "full_response": "", "elapsed_time": time.time() - start,
                          "tokens_input": 0, "tokens_output": 0, "cases_retrieved": 0,
                          "statutes_retrieved": 0, "status": "error", "error": str(e), "retry_count": attempt}

        with self.print_lock:
            self.completed_count += 1
            icon = "✓" if result["is_correct"] else "✗"
            print(f"[{self.completed_count}/{total}] #{idx} {result['category'][:6]} | "
                  f"정답:{result['expected_answer']} 추출:{result['extracted_answer']} {icon} | "
                  f"{result['elapsed_time']:.1f}s")
        return result

    def run_benchmark_batch(self, csv_path: str, batch_size: int, save_path: str, metadata: Dict) -> List[Dict]:
        all_data = self.load_benchmark_data(csv_path)
        total = len(all_data)
        completed = set()
        if save_path and os.path.exists(save_path):
            try:
                with open(save_path, "r", encoding="utf-8") as f:
                    self.results = json.load(f).get("results", [])
                    completed = {r["idx"] for r in self.results}
                    print(f"기존 결과 로드: {len(completed)}개 완료, 이어서 실행")
            except (json.JSONDecodeError, KeyError):
                self.results = []
        end = min(batch_size, total)
        data = all_data[:end]
        remaining = [(i, row) for i, row in enumerate(data) if (i + 1) not in completed]
        if not remaining:
            print("모든 문제가 이미 완료됨.")
            return self.results
        self.completed_count = len(completed)
        print(f"\n{'='*60}\nHybrid Naive RAG | {self.model} budget={self.thinking_budget} "
              f"granularity={self.granularity} top_k={self.top_k}\n남은 문제 {len(remaining)} / 전체 {total} | 워커 {self.num_workers}\n{'='*60}")
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures = {ex.submit(self.run_single_query, row["prompt"], row["category"],
                                 row["answer"], i + 1, total): i for i, row in remaining}
            for fut in as_completed(futures):
                r = fut.result()
                with self.results_lock:
                    self.results.append(r)
                    self.results.sort(key=lambda x: x.get("idx", 0))
                    self.save_results(save_path, metadata)
        self.results.sort(key=lambda x: x.get("idx", 0))
        return self.results

    def save_results(self, path: str, metadata: Optional[Dict] = None):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"metadata": metadata or {}, "results": self.results}, f, ensure_ascii=False, indent=2)

    def print_summary(self):
        if not self.results:
            print("결과 없음.")
            return
        total = len(self.results)
        correct = sum(1 for r in self.results if r.get("is_correct"))
        ti = sum(r["tokens_input"] for r in self.results)
        to = sum(r["tokens_output"] for r in self.results)
        ic = sum(calc_cost(self.model, r["tokens_input"], r["tokens_output"])[0] for r in self.results)
        oc = sum(calc_cost(self.model, r["tokens_input"], r["tokens_output"])[1] for r in self.results)
        print("\n" + "=" * 60)
        print(f"Hybrid Naive RAG 결과 | {self.model} budget={self.thinking_budget} granularity={self.granularity}")
        print("=" * 60)
        print(f"정답률: {correct}/{total} ({correct/total*100:.2f}%)")
        print(f"토큰: in {ti:,} / out {to:,} / 합 {ti+to:,}")
        print(f"비용: ${ic+oc:.4f}")
        print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="Gemini Hybrid Naive RAG 벤치마크 (rec1)")
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "max"])
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="Gemini 2.5계열 thinking_budget 직접 지정 (effort 무시, budget 스윕용. 예: 128 512 2048 8192 32768)")
    ap.add_argument("--granularity", default="full", choices=["summary", "full"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    thinking_budget = args.thinking_budget
    total = args.limit if args.limit else TOTAL_PROBLEMS
    csv_path = args.csv or os.path.join(os.path.dirname(__file__), "../benchmark_2025.csv")
    eff_slug = f"budget{thinking_budget}" if thinking_budget is not None else args.effort
    if args.output:
        out = args.output
    else:
        slug = args.model.replace("-", "_").replace(".", "_")
        out = os.path.join(os.path.dirname(__file__), "../results/2025",
                           f"rag_hybrid_{slug}_{eff_slug}_{args.granularity}_result.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    if is_legacy_model(args.model):
        _eff_budget = thinking_budget if thinking_budget is not None else EFFORT_TO_BUDGET[args.effort]
        thinking_info = f"thinking_budget={_eff_budget}"
    else:
        thinking_info = f"thinking_level={EFFORT_TO_LEVEL[args.effort]}"

    print(f"CSV: {csv_path}\n출력: {out}\nES: {ES_URL}\n{thinking_info} | granularity={args.granularity}")
    meta = {
        "experiment_name": "Hybrid Naive RAG Benchmark (rec1 controlled)",
        "model": args.model, "effort": args.effort, "thinking_budget": thinking_budget,
        "thinking_info": thinking_info, "granularity": args.granularity,
        "top_k": args.top_k, "n_full_cases": (N_FULL_CASES if args.granularity == "full" else 0),
        "case_full_max_chars": CASE_FULL_MAX, "case_summary_max_chars": CASE_SUMMARY_MAX,
        "retrieval": "hybrid (BM25+dense, RRF k=60 w1.05/1.0)",
        "case_index": INDEX_CASES, "statute_index": INDEX_STATUTES_V2,
        "case_bm25_fields": CASE_BM25_FIELDS, "statute_bm25_fields": STATUTE_BM25_FIELDS,
        "elasticsearch_url": ES_URL, "embedding_model": "embedding-query (Upstage)",
        "start_time": datetime.now().isoformat(), "total_problems": total,
    }
    bench = HybridRAGBenchmark(args.model, args.effort, args.granularity, args.workers, args.top_k, thinking_budget)
    results = bench.run_benchmark_batch(csv_path, total, out, meta)
    meta["end_time"] = datetime.now().isoformat()
    bench.results = results
    bench.save_results(out, meta)
    bench.print_summary()


if __name__ == "__main__":
    main()
