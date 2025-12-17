"""Search cases tool - hybrid search for court judgments"""
import logging
from typing import Dict, Any

from ..config import config
from ..elasticsearch.client import es_client
from ..elasticsearch.queries import query_builder
from ..utils.embedding import get_embedding
from ..utils.formatters import case_formatter
from ..utils.rrf_fusion import rrf_fusion

logger = logging.getLogger(__name__)


async def search_cases(
    query: str,
    reference_statute: str = None,
    court_name: str = None,
    date_from: str = None,
    date_to: str = None
) -> str:
    """
    Search Korean court cases (판례) by keywords, statutes, case types, or time periods.

    Results are ranked by hybrid relevance score combining:
    - BM25 keyword matching
    - Semantic similarity (embeddings)
    - Court level boost (대법원 > 고등법원 > 지방법원)
    - Recency boost (newer cases ranked higher)
    - Citation count boost (frequently cited cases ranked higher)

    When to use:
    - User asks for court judgments, precedents, or case law
    - User mentions specific statutes or legal issues
    - User wants to find similar past cases

    Args:
        query: Main search keywords (e.g., "개인정보보호법", "명예훼손")
               Keep it simple - just the core legal concepts or statute names.

        reference_statute: Filter by specific statute reference (optional)
                          Format: "법령명제조항" (e.g., "민법제911조", "형법제307조", "헌법제10조")
                          Exact matching - only returns cases citing this specific statute.

        court_name: Filter by specific court (optional)
                   - "대법원" (Supreme Court)
                   - "헌법재판소" (Constitutional Court)
                   - "고등법원" (High Court)
                   - "지방법원" (District Court)

        date_from: Start date filter in YYYYMMDD format (optional)
                  Example: "20200101" for cases from 2020-01-01 onwards

        date_to: End date filter in YYYYMMDD format (optional)
                Example: "20231231" for cases until 2023-12-31

    Returns:
        Markdown-formatted list of cases with:
        - Case number, name, court, date
        - Brief summary and referenced statutes
        - Citation count (how often this case is referenced)
        - Token count (length of full judgment text)

    Note: Results are sorted by relevance (hybrid search ranking).
    """
    # Use fixed top_k from config
    top_k = config.SEARCH_TOP_K

    try:
        logger.info(f"Searching cases: query='{query}', reference_statute={reference_statute}, "
                   f"court={court_name}, date_from={date_from}, date_to={date_to}, top_k={top_k}")

        # Step 1: Generate embedding for semantic search
        embedding = await get_embedding(query)
        logger.debug(f"Generated embedding (dim: {len(embedding)})")

        # Step 2: Build separate BM25 and vector queries for RRF
        logger.debug("Using RRF fusion mode")

        bm25_query = query_builder.build_bm25_only_case_query(
            query=query,
            reference_statute=reference_statute,
            court_name=court_name,
            date_from=date_from,
            date_to=date_to
        )

        vector_query = query_builder.build_vector_only_case_query(
            embedding=embedding,
            reference_statute=reference_statute,
            court_name=court_name,
            date_from=date_from,
            date_to=date_to
        )

        # Fetch more candidates for RRF
        fetch_size = max(top_k * 3, 50)

        # Execute BM25 search
        bm25_response = await es_client.search(
            index=config.INDEX_COURT_CASES,
            query=bm25_query,
            size=fetch_size
        )

        # Execute vector search
        vector_response = await es_client.search(
            index=config.INDEX_COURT_CASES,
            query=vector_query,
            size=fetch_size
        )

        # Fuse results using RRF
        bm25_hits = bm25_response["hits"]["hits"]
        vector_hits = vector_response["hits"]["hits"]

        logger.debug(f"BM25 candidates: {len(bm25_hits)}, Vector candidates: {len(vector_hits)}")

        fused_hits = rrf_fusion.fuse_elasticsearch_hits(
            bm25_hits=bm25_hits,
            vector_hits=vector_hits,
            k=config.RRF_K,
            bm25_weight=config.RRF_BM25_WEIGHT,
            vector_weight=config.RRF_VECTOR_WEIGHT
        )

        # Take top_k results
        hits = fused_hits[:top_k]
        logger.info(f"RRF fusion complete: {len(hits)} results from {len(fused_hits)} candidates")

        # Step 3: Format results
        formatted_output = case_formatter.format_search_results(hits)

        return formatted_output

    except Exception as e:
        logger.error(f"Error searching cases: {e}", exc_info=True)
        return f"Error searching cases: {str(e)}"
