"""Get statute content tool - retrieve full statute or specific articles"""
import logging
from typing import Optional

from ..config import config
from ..elasticsearch.client import es_client
from ..elasticsearch.queries import query_builder
from ..utils.formatters import statute_formatter

logger = logging.getLogger(__name__)


async def get_statute_content(
    statute_id: str,
    article_number: Optional[str] = None,
    article_range: Optional[str] = None
) -> str:
    """
    Retrieve full statute text or specific articles by statute ID.

    Args:
        statute_id: Statute ID from search results
                   Examples: "1706", "1692", "20547"

        article_number: Get one article (e.g., "15", "750")
                       Examples: "15" → Article 15 only

        article_range: Get multiple articles (e.g., "1-10", "750-760")
                      Examples: "1-10" → Articles 1 through 10

    Returns:
        Statute text with:
        - Law metadata (effective date, type)
        - Full text of requested articles
        - Citation count (how often used in court cases)

    Note: Cannot use both article_number and article_range together.
    """
    # Validate parameters
    if article_number and article_range:
        return "Error: Cannot specify both article_number and article_range. Use only one."

    try:
        logger.info(f"Retrieving statute: {statute_id}, "
                   f"article_number: {article_number}, article_range: {article_range}")

        # Build query
        es_query = query_builder.build_statute_content_query(
            statute_id=statute_id,
            article_number=article_number,
            article_range=article_range
        )

        # Execute search
        response = await es_client.search(
            index=config.INDEX_STATUTES,
            query=es_query,
            size=1000  # Max articles to retrieve
        )

        hits = response["hits"]["hits"]

        if not hits:
            logger.warning(f"Statute {statute_id} not found")
            return f"Statute '{statute_id}' not found in the database."

        # Extract clause data
        clauses = [hit["_source"] for hit in hits]

        # Get metadata from first clause
        first_clause = clauses[0]
        law_name = first_clause.get("law_name", "N/A")
        law_type = first_clause.get("law_type", "N/A")
        effective_date = first_clause.get("effective_date", "N/A")
        promulgation_date = first_clause.get("promulgation_date", "N/A")
        law_detail_link = first_clause.get("law_detail_link", "")

        # Get total clause count and overall citation count from metadata index
        metadata = await es_client.get_statute_metadata(statute_id)

        total_clauses = len(clauses)  # Default to actual clause count
        total_citation_count = 0  # 법령 전체 인용 횟수
        abbreviation = ""

        if metadata:
            total_clauses = metadata.get("clause_count", len(clauses))
            total_citation_count = metadata.get("reference_case_count", 0)
            abbreviation = metadata.get("abbreviation", "")

        logger.info(f"Retrieved {len(clauses)} clauses from statute {law_name}")

        # Format output
        formatted_output = statute_formatter.format_statute_content(
            statute_id=statute_id,
            law_name=law_name,
            law_type=law_type,
            effective_date=effective_date,
            promulgation_date=promulgation_date,
            total_clauses=total_clauses,
            total_citation_count=total_citation_count,  # 법령 전체 인용 횟수
            abbreviation=abbreviation,
            law_detail_link=law_detail_link,
            clauses=clauses,
            article_number=article_number,
            article_range=article_range
        )

        return formatted_output

    except Exception as e:
        logger.error(f"Error retrieving statute content: {e}", exc_info=True)
        return f"Error retrieving statute content: {str(e)}"
