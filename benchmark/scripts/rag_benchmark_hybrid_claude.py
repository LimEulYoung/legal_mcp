#!/usr/bin/env python3
"""
Claude Naive RAG (HYBRID retrieval) 벤치마크 스크립트  [rec1 통제실험용]

rag_benchmark_hybrid_gpt.py와 검색/granularity 로직 동일, LLM 호출만 Anthropic으로 교체.

설계 (rec1 교란 통제) — GPT 하이브리드와 100% 동일:
  - 판례:  test_court_cases_new — 에이전트와 '동일한' BM25 필드 부스트 + KNN(embedding_vector) → RRF
  - 법령:  test_statutes_v2 (조문 단위) — '균등' BM25(clause_content/clause_title/law_name) + KNN(embedding) → RRF
  - RRF:   k=60, bm25_weight=1.05, vector_weight=1.0
  - top_k: 10
  - granularity: summary -> 요약 10건 / full -> 요약 10건 + 상위 5건 전문(5000자)

Claude thinking:
  effort → budget_tokens: none=0(disabled,temp0), low=1024, medium=8192, high=32768, max=63999 (enabled,temp1)
  논문 구성 = Non-Thinking(none) + Max-Thinking(max)

사용 예시:
  python rag_benchmark_hybrid_claude.py --model claude-sonnet-4-5-20250929 --effort max --granularity full --workers 3
  python rag_benchmark_hybrid_claude.py --model claude-sonnet-4-5-20250929 --effort none --granularity summary --limit 10
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
from openai import OpenAI          # Upstage 임베딩용
from elasticsearch import Elasticsearch
import anthropic

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

ES_HOST = os.environ.get("ES_HOST") or "your-es-host"
ES_PORT = os.environ.get("ES_PORT") or "9200"
ES_SCHEME = os.environ.get("ES_SCHEME") or "http"
ES_USER = os.environ.get("ES_USER") or "elastic"
ES_PASSWORD = os.environ.get("ES_PASSWORD") or "your-es-password"
ES_URL = f"{ES_SCHEME}://{ES_HOST}:{ES_PORT}"
UPSTAGE_API_KEY = os.environ.get("UPSTAGE_API_KEY") or "your-upstage-api-key"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY") or "your-anthropic-api-key"

# 인덱스 (논문 동결본)
INDEX_CASES = "test_court_cases_new"
INDEX_STATUTES_V2 = "test_statutes_v2"

# RRF / KNN (에이전트 config와 동일)
RRF_K = 60; RRF_BM25_WEIGHT = 1.05; RRF_VECTOR_WEIGHT = 1.0
KNN_K = 30; KNN_NUM_CANDIDATES = 150

# 판례 BM25 필드 부스트 (에이전트 config 값 동결)
CASE_BM25_FIELDS = [
    "case_number^1.5", "case_name^0.8", "reference_statute^3.5",
    "judged_statute^3.5", "judgment_summary^6.0", "case_content^2.5",
]
# 법령 BM25 필드 (균등 = 튜닝 없음)
STATUTE_BM25_FIELDS = ["clause_content", "clause_title", "law_name"]

CASE_SUMMARY_MAX = 2000
CASE_FULL_MAX = 5000
N_FULL_CASES = 5

# Claude effort → thinking budget_tokens
EFFORT_TO_THINKING_TOKENS = {"none": 0, "low": 1024, "medium": 8192, "high": 32768, "max": 63999}

TOTAL_PROBLEMS = 150
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

MODEL_PRICING = {
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
}
DEFAULT_PRICING = (3.00, 15.00)


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> tuple:
    ip, op = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return (input_tokens / 1_000_000) * ip, (output_tokens / 1_000_000) * op


def rrf_fuse(bm25_hits, vector_hits, k=RRF_K, bm25_weight=RRF_BM25_WEIGHT, vector_weight=RRF_VECTOR_WEIGHT):
    """src/utils/rrf_fusion.py와 동일한 순위기반 RRF 융합."""
    doc = defaultdict(lambda: {"rrf": 0.0, "hit": None})
    for rank, hit in enumerate(bm25_hits):
        d = doc[hit["_id"]]; d["rrf"] += (1.0 / (rank + k)) * bm25_weight; d["hit"] = hit
    for rank, hit in enumerate(vector_hits):
        d = doc[hit["_id"]]; d["rrf"] += (1.0 / (rank + k)) * vector_weight
        if d["hit"] is None: d["hit"] = hit
    fused = []
    for _id, data in sorted(doc.items(), key=lambda x: x[1]["rrf"], reverse=True):
        h = dict(data["hit"]); h["_score"] = data["rrf"]; fused.append(h)
    return fused


class ClaudeHybridRAGBenchmark:
    def __init__(self, model: str, effort: str, granularity: str, num_workers: int = 3, top_k: int = 10):
        self.model = model
        self.effort = effort
        self.granularity = granularity
        self.top_k = top_k
        self.es = Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASSWORD), request_timeout=60)
        self.upstage = OpenAI(api_key=UPSTAGE_API_KEY, base_url="https://api.upstage.ai/v1")
        self.results = []
        self.num_workers = num_workers
        self.results_lock = threading.Lock()
        self.print_lock = threading.Lock()
        self.completed_count = 0

    def get_embedding(self, text: str) -> List[float]:
        return self.upstage.embeddings.create(input=text, model="embedding-query").data[0].embedding

    # ---------------- HYBRID 검색 (GPT 하이브리드와 동일) ----------------
    def _search_cases_hybrid(self, query_text, query_vector):
        fetch = max(self.top_k * 3, 50)
        bm25 = self.es.search(index=INDEX_CASES, size=fetch, body={
            "query": {"bool": {"must": [{"multi_match": {"query": query_text, "fields": CASE_BM25_FIELDS, "type": "best_fields"}}]}},
            "_source": {"excludes": ["case_content", "embedding_vector"]}})["hits"]["hits"]
        vec = self.es.search(index=INDEX_CASES, size=fetch, body={
            "knn": {"field": "embedding_vector", "query_vector": query_vector, "k": KNN_K, "num_candidates": KNN_NUM_CANDIDATES},
            "_source": {"excludes": ["case_content", "embedding_vector"]}})["hits"]["hits"]
        return rrf_fuse(bm25, vec)[:self.top_k]

    def _search_statutes_hybrid(self, query_text, query_vector):
        fetch = max(self.top_k * 3, 50)
        src = ["law_name", "clause_number", "clause_title", "clause_content", "law_type", "effective_date", "reference_case_count"]
        bm25 = self.es.search(index=INDEX_STATUTES_V2, size=fetch, body={
            "query": {"bool": {"must": [{"multi_match": {"query": query_text, "fields": STATUTE_BM25_FIELDS, "type": "best_fields"}}]}},
            "_source": src})["hits"]["hits"]
        vec = self.es.search(index=INDEX_STATUTES_V2, size=fetch, body={
            "knn": {"field": "embedding", "query_vector": query_vector, "k": KNN_K, "num_candidates": KNN_NUM_CANDIDATES},
            "_source": src})["hits"]["hits"]
        return rrf_fuse(bm25, vec)[:self.top_k]

    def _fetch_case_contents(self, ids):
        if not ids: return {}
        resp = self.es.mget(body={"docs": [{"_index": INDEX_CASES, "_id": i, "_source": ["case_content"]} for i in ids]})
        return {d["_id"]: (d["_source"] or {}).get("case_content", "") or "" for d in resp["docs"] if d.get("found")}

    # ---------------- context 포맷 (요약 10 + full모드 상위 5건 전문) ----------------
    def format_cases_context(self, cases, full_map):
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
            if len(summary) > CASE_SUMMARY_MAX: summary = summary[:CASE_SUMMARY_MAX] + "..."
            lines.append(f"    - judgment_summary: {summary}")
            if self.granularity == "full" and rank < N_FULL_CASES:
                body = full_map.get(hit["_id"], "") or ""
                if body:
                    if len(body) > CASE_FULL_MAX: body = body[:CASE_FULL_MAX] + "..."
                    lines.append(f"    - judgment_text: {body}")
            lines.append(f"    - reference_statutes: {src.get('reference_statute', 'N/A')}")
            lines.append(f"    - citation_count: {src.get('reference_case_count', 0)}")
            lines.append(f"    - relevance_score: {hit.get('_score', 0):.4f}")
            tc = src.get("token_count")
            lines.append(f"    - token_count: {tc:,}" if tc else "    - token_count: N/A")
        lines.append("    ----------")
        return "\n".join(lines)

    def format_statutes_context(self, statutes):
        lines = ["Available statutes (top matches):", """
Each result includes:
- law_name, clause_number, clause_title, clause_content
- law_type, effective_date, citation_count, relevance_score
"""]
        for hit in statutes:
            src = hit["_source"]
            eff = src.get("effective_date", "")
            eff_str = (f"{str(eff)[:4]}-{str(eff)[4:6]}-{str(eff)[6:8]}" if eff and len(str(eff)) == 8 else (str(eff) or "N/A"))
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

    # ---------------- LLM 호출 (Anthropic + thinking, rag_benchmark_claude.py와 동일) ----------------
    def ask_claude(self, prompt: str, context: str) -> Dict:
        formatted_prompt = f"""{context}

---

{prompt}

**중요: 반드시 다음 형식으로 최종 답변을 제시하세요:**
answer: [1-5 중 하나의 숫자]"""
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        budget = EFFORT_TO_THINKING_TOKENS.get(self.effort, 32768)
        params = {"model": self.model, "max_tokens": 64000,
                  "messages": [{"role": "user", "content": formatted_prompt}]}
        if budget == 0:
            params["temperature"] = 0
            params["thinking"] = {"type": "disabled"}
        else:
            params["temperature"] = 1
            params["thinking"] = {"type": "enabled", "budget_tokens": budget}
        with client.messages.stream(**params) as stream:
            response = stream.get_final_message()
        text = ""; thinking = ""
        for block in response.content:
            if block.type == "thinking": thinking += block.thinking
            elif block.type == "text": text += block.text
        return {"output_text": text, "thinking_text": thinking,
                "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
                "formatted_prompt": formatted_prompt}

    def load_benchmark_data(self, csv_path):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def extract_answer(self, t):
        for p in [r'answer:\s*\*\*([1-5])\*\*', r'answer:\s*([1-5])', r'정답:\s*([1-5])',
                  r'Answer:\s*([1-5])', r'ANSWER:\s*([1-5])', r'답:\s*([1-5])']:
            m = re.search(p, t)
            if m: return m.group(1)
        sym = {'①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5'}
        for s, n in sym.items():
            if f"정답: {s}" in t or f"정답은 {s}" in t: return n
        return None

    def grade_answer(self, ext, exp):
        return ext is not None and str(ext).strip() == str(exp).strip()

    def run_single_query(self, prompt, category, expected, idx=0, total=0, max_retries=5):
        result = None
        for attempt in range(max_retries):
            start = time.time()
            try:
                qv = self.get_embedding(prompt)
                s = time.time()
                cases = self._search_cases_hybrid(prompt, qv)
                statutes = self._search_statutes_hybrid(prompt, qv)
                full_map = self._fetch_case_contents([h["_id"] for h in cases[:N_FULL_CASES]]) if self.granularity == "full" else {}
                search_time = time.time() - s
                ctx = f"{self.format_cases_context(cases, full_map)}\n\n{self.format_statutes_context(statutes)}"
                l = time.time()
                cl = self.ask_claude(prompt, ctx)
                llm_time = time.time() - l
                text = cl["output_text"]; ext = self.extract_answer(text); ok = self.grade_answer(ext, expected)
                if attempt < max_retries - 1 and (not text or ext is None):
                    w = (2 ** attempt) * 10
                    with self.print_lock: print(f"\n[{idx}/{total}] 응답/파싱 실패, {w}s 후 재시도 ({attempt+1}/{max_retries})")
                    time.sleep(w); continue
                result = {
                    "idx": idx, "category": category, "full_prompt": cl["formatted_prompt"],
                    "expected_answer": expected, "extracted_answer": ext, "is_correct": ok,
                    "full_response": text, "full_thinking": cl["thinking_text"],
                    "elapsed_time": time.time() - start, "search_time": search_time, "llm_time": llm_time,
                    "tokens_input": cl["input_tokens"], "tokens_output": cl["output_tokens"],
                    "cases_retrieved": len(cases), "statutes_retrieved": len(statutes),
                    "retrieved_cases": [{"case_number": h["_source"].get("case_number"), "score": round(h.get("_score", 0), 6)} for h in cases],
                    "retrieved_statutes": [{"law_name": h["_source"].get("law_name"), "clause_number": h["_source"].get("clause_number"), "score": round(h.get("_score", 0), 6)} for h in statutes],
                    "status": "success", "retry_count": attempt}
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    w = (2 ** attempt) * 10
                    with self.print_lock: print(f"\n[{idx}/{total}] 에러, {w}s 후 재시도 ({attempt+1}/{max_retries}): {str(e)[:120]}")
                    time.sleep(w); continue
                result = {"idx": idx, "category": category, "full_prompt": prompt, "expected_answer": expected,
                          "extracted_answer": None, "is_correct": False, "full_response": "", "full_thinking": "",
                          "elapsed_time": time.time() - start, "tokens_input": 0, "tokens_output": 0,
                          "cases_retrieved": 0, "statutes_retrieved": 0, "status": "error", "error": str(e)[:500], "retry_count": attempt}
        with self.print_lock:
            self.completed_count += 1
            icon = "✓" if result["is_correct"] else "✗"
            print(f"[{self.completed_count}/{total}] #{idx} {result['category'][:6]} | 정답:{result['expected_answer']} 추출:{result['extracted_answer']} {icon} | {result['elapsed_time']:.1f}s")
        return result

    def run_benchmark_batch(self, csv_path, batch_size, save_path, metadata):
        all_data = self.load_benchmark_data(csv_path); total = len(all_data); completed = set()
        if save_path and os.path.exists(save_path):
            try:
                with open(save_path, "r", encoding="utf-8") as f:
                    self.results = json.load(f).get("results", []); completed = {r["idx"] for r in self.results}
                    print(f"기존 결과 로드: {len(completed)}개 완료, 이어서 실행")
            except (json.JSONDecodeError, KeyError): self.results = []
        data = all_data[:min(batch_size, total)]
        remaining = [(i, row) for i, row in enumerate(data) if (i + 1) not in completed]
        if not remaining: print("모든 문제가 이미 완료됨."); return self.results
        self.completed_count = len(completed)
        print(f"\n{'='*60}\nClaude Hybrid Naive RAG | {self.model} effort={self.effort} (budget={EFFORT_TO_THINKING_TOKENS.get(self.effort)}) granularity={self.granularity}\n남은 {len(remaining)}/{total} | 워커 {self.num_workers}\n{'='*60}")
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futs = {ex.submit(self.run_single_query, row["prompt"], row["category"], row["answer"], i + 1, total): i for i, row in remaining}
            for fut in as_completed(futs):
                r = fut.result()
                with self.results_lock:
                    self.results.append(r); self.results.sort(key=lambda x: x.get("idx", 0)); self.save_results(save_path, metadata)
        self.results.sort(key=lambda x: x.get("idx", 0))
        return self.results

    def save_results(self, path, metadata=None):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"metadata": metadata or {}, "results": self.results}, f, ensure_ascii=False, indent=2)

    def print_summary(self):
        if not self.results: print("결과 없음."); return
        total = len(self.results); correct = sum(1 for r in self.results if r.get("is_correct"))
        ti = sum(r["tokens_input"] for r in self.results); to = sum(r["tokens_output"] for r in self.results)
        cost = sum(sum(calc_cost(self.model, r["tokens_input"], r["tokens_output"])) for r in self.results)
        print("\n" + "=" * 60)
        print(f"Claude Hybrid Naive RAG | {self.model} effort={self.effort} granularity={self.granularity}")
        print(f"정답률: {correct}/{total} ({100*correct/total:.2f}%) | 토큰 in {ti:,}/out {to:,} | ${cost:.4f}")
        print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="Claude Hybrid Naive RAG 벤치마크 (rec1)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default="max", choices=["none", "low", "medium", "high", "max"])
    ap.add_argument("--granularity", default="full", choices=["summary", "full"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    total = args.limit if args.limit else TOTAL_PROBLEMS
    csv_path = args.csv or os.path.join(os.path.dirname(__file__), "../benchmark_2025.csv")
    if args.output:
        out = args.output
    else:
        slug = args.model.replace("-", "_").replace(".", "_")
        lab = args.effort if args.effort != "none" else "non_thinking"
        out = os.path.join(os.path.dirname(__file__), "../results/2025", f"rag_hybrid_{slug}_{lab}_{args.granularity}_result.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    print(f"CSV: {csv_path}\n출력: {out}\nES: {ES_URL}")
    meta = {
        "experiment_name": "Claude Hybrid Naive RAG (rec1 controlled)",
        "model": args.model, "effort": args.effort, "thinking_budget_tokens": EFFORT_TO_THINKING_TOKENS.get(args.effort),
        "granularity": args.granularity, "top_k": args.top_k,
        "n_full_cases": (N_FULL_CASES if args.granularity == "full" else 0),
        "case_full_max_chars": CASE_FULL_MAX, "case_summary_max_chars": CASE_SUMMARY_MAX,
        "retrieval": "hybrid (BM25+dense, RRF k=60 w1.05/1.0)",
        "case_index": INDEX_CASES, "statute_index": INDEX_STATUTES_V2,
        "case_bm25_fields": CASE_BM25_FIELDS, "statute_bm25_fields": STATUTE_BM25_FIELDS,
        "elasticsearch_url": ES_URL, "embedding_model": "embedding-query (Upstage)",
        "start_time": datetime.now().isoformat(), "total_problems": total}
    b = ClaudeHybridRAGBenchmark(args.model, args.effort, args.granularity, args.workers, args.top_k)
    results = b.run_benchmark_batch(csv_path, total, out, meta)
    meta["end_time"] = datetime.now().isoformat()
    b.results = results; b.save_results(out, meta); b.print_summary()


if __name__ == "__main__":
    main()
