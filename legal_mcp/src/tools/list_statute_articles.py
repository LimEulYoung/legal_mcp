"""List statute articles tool - retrieve table of contents with titles only"""
import logging
from typing import Optional, Dict, Any

from ..config import config
from ..elasticsearch.client import es_client
from ..elasticsearch.queries import query_builder
from ..utils.formatters import statute_formatter

logger = logging.getLogger(__name__)


async def list_statute_articles(statute_id: str) -> str:
    """
    List all articles (목차) of a statute with titles only.

    Perfect for exploring large statutes (e.g., 민법 with 1,193 articles)
    when you don't know the specific article number.

    When to use:
    - User wants to browse statute structure
    - User wants to see table of contents (e.g., "민법 조문 목차 보여줘")
    - User needs to find article numbers before reading full text
    - After search_statutes(), when user wants to see what's inside

    Args:
        statute_id: ID from search_statutes
                   Examples: "1706" (민법), "1692" (형법), "20547" (개인정보보호법)

    Returns:
        Table of contents with:
        - Article number (e.g., 제750조)
        - Article title (e.g., 불법행위의 내용)
        - Citation count (how often each article is cited in court cases)

    Note: Only returns titles. Use get_statute_content() to read full article text.
    """
    try:
        logger.info(f"Listing articles for statute: {statute_id}")

        # Build query
        es_query = query_builder.build_statute_articles_list_query(
            statute_id=statute_id
        )

        # Execute search
        response = await es_client.search(
            index=config.INDEX_STATUTES,
            query=es_query,
            size=2000  # Max articles to retrieve (민법 has 1,193)
        )

        hits = response["hits"]["hits"]

        if not hits:
            logger.warning(f"Statute {statute_id} not found")
            return f"Statute '{statute_id}' not found in the database."

        # Extract article data
        articles = [hit["_source"] for hit in hits]

        # Get metadata from first article
        first_article = articles[0]
        law_name = first_article.get("law_name", "N/A")

        # Get total clause count from metadata index
        metadata = await es_client.get_statute_metadata(statute_id)

        total_clauses = len(articles)  # Default to actual article count
        abbreviation = ""

        if metadata:
            total_clauses = metadata.get("clause_count", len(articles))
            abbreviation = metadata.get("abbreviation", "")

        logger.info(f"Retrieved {len(articles)} articles from statute {law_name}")

        # Format output
        formatted_output = statute_formatter.format_articles_list(
            statute_id=statute_id,
            law_name=law_name,
            abbreviation=abbreviation,
            total_clauses=total_clauses,
            articles=articles
        )

        return formatted_output

    except Exception as e:
        logger.error(f"Error listing statute articles: {e}", exc_info=True)
        return f"Error listing statute articles: {str(e)}"
