"""Search statutes tool - hybrid search for statute metadata"""
import logging
from typing import Dict, Any

from ..config import config
from ..elasticsearch.client import es_client
from ..elasticsearch.queries import query_builder
from ..utils.embedding import get_embedding
from ..utils.formatters import statute_formatter
from ..utils.rrf_fusion import rrf_fusion

logger = logging.getLogger(__name__)


async def search_statutes(
    query: str,
    law_type: str = None
) -> str:
    """
    Search Korean statutes by law name or abbreviation.

    When to use:
    - User asks for specific laws or regulations (e.g., "개인정보보호법", "민법")
    - User wants to find statutes by topic or abbreviation
    - Starting point to get statute_id before reading articles with get_statute_content()
      or browsing table of contents with list_statute_articles()

    Args:
        query: Law name/abbreviation (e.g., "개인정보보호법", "민법")
               Simple keywords work best - just the law name or topic.

        law_type: Filter by type (optional)
                 Examples: "법률", "대통령령", "대법원규칙", "국토교통부령",
                          "보건복지부령", "총리령" 등 (any valid law type)
                 Leave empty to search all types.

    Returns:
        Statutes with statute_id, law_name, abbreviation, law_type, clause_count,
        description (Article 1), citation_count, token_count.
        Sorted by relevance (hybrid search ranking).

    Note: Use statute_id from results to read articles or browse table of contents.
    """
    # Use fixed top_k from config
    top_k = config.SEARCH_TOP_K

    try:
        logger.info(f"Searching statutes: query='{query}', law_type={law_type}, top_k={top_k}")

        # Step 1: Generate embedding for semantic search
        embedding = await get_embedding(query)
        logger.debug(f"Generated embedding (dim: {len(embedding)})")

        # Step 2: Build separate BM25 and vector queries for RRF
        logger.debug("Using RRF fusion mode")

        bm25_query = query_builder.build_bm25_only_statute_query(
            query=query,
            law_type=law_type
        )

        vector_query = query_builder.build_vector_only_statute_query(
            embedding=embedding,
            law_type=law_type
        )

        # Fetch more candidates for RRF
        fetch_size = max(top_k * 3, 50)

        # Execute BM25 search
        bm25_response = await es_client.search(
            index=config.INDEX_STATUTES_METADATA,
            query=bm25_query,
            size=fetch_size
        )

        # Execute vector search
        vector_response = await es_client.search(
            index=config.INDEX_STATUTES_METADATA,
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
        formatted_output = statute_formatter.format_search_results(hits)

        return formatted_output

    except Exception as e:
        logger.error(f"Error searching statutes: {e}", exc_info=True)
        return f"Error searching statutes: {str(e)}"
