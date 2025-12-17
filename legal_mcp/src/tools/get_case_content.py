"""Get case content tool - retrieve full judgment text"""
import logging

from ..config import config
from ..elasticsearch.client import es_client
from ..utils.formatters import case_formatter

logger = logging.getLogger(__name__)


async def get_case_content(case_number: str) -> str:
    """
    Retrieve the full text of a specific court case.

    When to use:
    - User wants to read the complete judgment text
    - User asks "show me the full case" or "read the entire decision"
    - After search_cases(), when user selects a specific case number

    Args:
        case_number: Case ID from search results (e.g., "2013누26042")

    Returns:
        Complete judgment text including:
        - Case metadata (court, date, judges)
        - Facts, reasoning, and conclusion
        - Referenced statutes and precedents

    Note: Very long text (up to 50,000 tokens). Use summarization if needed.
    """
    try:
        logger.info(f"Retrieving case content for: {case_number}")

        # Search by case_number field
        cases = await es_client.search_by_field(
            index=config.INDEX_COURT_CASES,
            field="case_number",
            value=case_number,
            size=1
        )

        if not cases:
            logger.warning(f"Case {case_number} not found")
            return f"Case '{case_number}' not found in the database."

        case = cases[0]
        logger.info(f"Retrieved case: {case.get('case_name', 'N/A')}")

        # Format output
        formatted_output = case_formatter.format_case_content(case)

        return formatted_output

    except Exception as e:
        logger.error(f"Error retrieving case content: {e}", exc_info=True)
        return f"Error retrieving case content: {str(e)}"
