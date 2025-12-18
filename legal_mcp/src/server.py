"""Legal Search MCP Server - Main entry point"""
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from fastmcp import FastMCP

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

from .config import config
from .tools.search_cases import search_cases
from .tools.search_statutes import search_statutes
from .tools.get_case_content import get_case_content
from .tools.get_statute_content import get_statute_content
from .tools.list_statute_articles import list_statute_articles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastMCP server
# Use "legal_mcp" as server name to match client's tool call format
mcp = FastMCP("legal_mcp")


# Register tools
@mcp.tool(name="search_cases")
async def search_cases_tool(
    query: str,
    reference_statute: str = None,
    court_name: str = None,
    date_from: str = None,
    date_to: str = None
) -> str:
    """
    Search Korean court cases (판례) by keywords, statutes, or legal topics.

    Results are ranked by hybrid relevance combining BM25 and semantic similarity.

    Args:
        query: Core legal concept or topic
               Examples: "개인정보보호법", "명예훼손", "불법행위", "손해배상"
               Keep simple - use filter parameters for court, dates, statutes

        reference_statute: Filter by statute citation (optional)
                          Format: "법령명제조항" (e.g., "민법제750조", "형법제307조")
                          Exact matching only

        court_name: Filter by court (optional)
                   Examples: "대법원", "헌법재판소", "서울고등법원", "서울행정법원"
                   Also supports specific district/high courts

        date_from: Start date in YYYYMMDD (optional, e.g., "20200101")

        date_to: End date in YYYYMMDD (optional, e.g., "20231231")

    Returns:
        Markdown list with case_number, case_name, court_name, decision_date,
        judgment_summary, reference_statutes, reference_cases, citation_count, token_count

    Examples:
        User: "2020년 이후 대법원 개인정보보호법 판례"
        → search_cases(query="개인정보보호법", court_name="대법원", date_from="20200101")

        User: "명예훼손 손해배상 판례"
        → search_cases(query="명예훼손 손해배상")

        User: "헌법재판소 위헌 결정"
        → search_cases(query="위헌", court_name="헌법재판소")

        User: "민법 제750조를 인용한 불법행위 판례"
        → search_cases(query="불법행위", reference_statute="민법제750조")
    """
    return await search_cases(
        query=query,
        reference_statute=reference_statute,
        court_name=court_name,
        date_from=date_from,
        date_to=date_to
    )


@mcp.tool(name="get_case_content")
async def get_case_content_tool(case_number: str) -> str:
    """
    Retrieve full judgment text by case number.

    Use this after search_cases() to get the complete judgment text.

    Args:
        case_number: Case identifier from search results (e.g., "2013누26042")

    Returns:
        Complete judgment with case_number, case_name, court_name, decision_date,
        reference_statute, and full case_content text.

    Note: Some judgments are very long (up to 50,000 tokens).
    """
    return await get_case_content(case_number)


@mcp.tool(name="search_statutes")
async def search_statutes_tool(
    query: str,
    law_type: str = None
) -> str:
    """
    Search Korean statutes (법령) by name, topic, or legal concept.

    Use this to find statute_id values before reading articles or browsing table of contents.
    Results are ranked by hybrid relevance combining BM25 and semantic similarity.

    Args:
        query: Statute name or legal concept
               Examples: "제조물책임법", "부정경쟁방지", "부당이득"

        law_type: Filter by law type (optional)
                 Examples: "법률", "대통령령", "대법원규칙", "국토교통부령",
                          "보건복지부령", "총리령", "헌법재판소규칙" 등
                 Supports any valid law type from the system.
                 Leave empty to search all types.

    Returns:
        Markdown list with statute_id, law_name, abbreviation, law_type,
        clause_count, description (Article 1), citation_count, token_count

    Examples:
        User: "제조물책임법"
        → search_statutes(query="제조물책임법")

        User: "부정경쟁방지 관련 법률"
        → search_statutes(query="부정경쟁방지", law_type="법률")

        User: "대법원규칙 중에 소송 관련"
        → search_statutes(query="소송", law_type="대법원규칙")
    """
    return await search_statutes(
        query=query,
        law_type=law_type
    )


@mcp.tool(name="get_statute_content")
async def get_statute_content_tool(
    statute_id: str,
    article_number: str = None,
    article_range: str = None
) -> str:
    """
    Retrieve statute content - full statute or specific articles.

    **Quick Access for Major Statutes**:
    헌법(1444), 민법(1706), 상법(1702), 민사소송법(1700), 형법(1692), 형사소송법(1671), 행정기본법(14041), 행정절차법(1362), 행정소송법(1363), 헌법재판소법(11233)

    Args:
        statute_id: From search results or Quick Access list

        article_number: Specific article (e.g., "15", "750") - optional

        article_range: Article range (e.g., "1-10", "750-760") - optional
                      Cannot use both article_number and article_range

    Returns:
        Statute text with law_name, law_type, effective_date, promulgation_date,
        total_clauses, total_citation_count, and full article content

    Examples:
        - get_statute_content("1706", article_number="750")  # 민법 제750조
        - get_statute_content("1444", article_range="10-11")   # 헌법 제10조, 제11조
        - get_statute_content("1692")  # 형법 전체
    """
    return await get_statute_content(statute_id, article_number, article_range)


@mcp.tool(name="list_statute_articles")
async def list_statute_articles_tool(statute_id: str) -> str:
    """
    List all articles (목차) of a statute with titles and citation counts.

    Use to browse statute structure or find specific article numbers in large statutes.

    Args:
        statute_id: From search results (e.g., "1706" for 민법)

    Returns:
        Table of contents with article numbers, titles, citation counts

    Note: Only titles. Use get_statute_content() for full article text.

    Examples:
        - User: "민법 조문 목차 보여줘" → list_statute_articles("1706")
        - User: "헌법 구조 알려줘" → list_statute_articles("1444")
        - User: "행정소송법 어떤 조항들 있어?" → list_statute_articles("1363")
    """
    return await list_statute_articles(statute_id)




def main():
    """Run the MCP server"""
    # Validate configuration on startup
    try:
        config.validate()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise

    # Determine transport mode from environment
    transport_mode = os.getenv("MCP_TRANSPORT", "stdio")

    if transport_mode == "sse":
        # Run with SSE transport for web deployment
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "8000"))
        logger.info(f"Starting Legal Search MCP Server with SSE transport on {host}:{port}...")
        mcp.run(transport="sse", host=host, port=port)
    else:
        # Run with stdio transport (default for local use)
        logger.info("Starting Legal Search MCP Server with stdio transport...")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
