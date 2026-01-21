"""Output formatters for MCP responses"""
from typing import List, Dict, Any, Optional

from ..config import config


class CaseFormatter:
    """Format case search results for MCP output"""

    @staticmethod
    def format_search_results(hits: List[Dict[str, Any]]) -> str:
        """
        Format case search results in markdown.

        Returns formatted string matching the specification in function_design.md
        """
        if not hits:
            return "No cases found matching your query."

        output = ["Available judgments (top matches):\n"]
        output.append("Each result includes:")
        output.append("- case_number: The unique identifier of the case")
        output.append("- case_name: The title or key issue of the case")
        output.append("- court_name: The name of the court that delivered the judgment")
        output.append("- decision_date: The date the judgment was rendered (YYYY-MM-DD)")
        output.append("- judgment_summary: A brief summary of the judgment")
        output.append("- reference_statutes: The legal provisions applied in the judgment")
        output.append("- reference_cases: The precedents cited in this judgment")
        output.append("- citation_count: Number of times this case has been cited by other cases")
        output.append("- token_count: Length of the full judgment text in tokens\n")
        output.append("Choose cases considering case name, court, citation count, summary, "
                     "referenced statutes, and overall relevance to your use case.\n")

        for hit in hits:
            source = hit["_source"]
            score = hit["_score"]

            output.append("    ----------")
            output.append(f"    - case_number: {source.get('case_number', 'N/A')}")
            output.append(f"    - case_name: {source.get('case_name', 'N/A')}")
            output.append(f"    - court_name: {source.get('court_name', 'N/A')}")

            # Format date
            decision_date = source.get('decision_date', '')
            if decision_date and len(decision_date) == 8:
                formatted_date = f"{decision_date[:4]}-{decision_date[4:6]}-{decision_date[6:]}"
            else:
                formatted_date = decision_date or 'N/A'
            output.append(f"    - decision_date: {formatted_date}")

            summary = source.get('judgment_summary', 'N/A')
            if summary and len(summary) > config.CASE_SUMMARY_MAX_LENGTH:
                summary = summary[:config.CASE_SUMMARY_MAX_LENGTH] + "..."
            output.append(f"    - judgment_summary: {summary}")

            output.append(f"    - reference_statutes: {source.get('reference_statute', 'N/A')}")
            output.append(f"    - reference_cases: {source.get('reference_case', 'N/A')}")

            # Add citation count (인용 횟수)
            citation_count = source.get('reference_case_count', 0)
            output.append(f"    - citation_count: {citation_count}")

            output.append(f"    - token_count: {source.get('token_count', 0):,}")

        output.append("    ----------")
        return "\n".join(output)

    @staticmethod
    def format_case_content(case: Dict[str, Any]) -> str:
        """
        Format full case content in markdown.

        Returns formatted string matching the specification in function_design.md
        """
        if not case:
            return "Case not found."

        output = ["=" * 16]
        output.append(f"CASE: {case.get('case_number', 'N/A')}")
        output.append("=" * 16)
        output.append("")
        output.append(f"Title: {case.get('case_name', 'N/A')}")
        output.append(f"Court: {case.get('court_name', 'N/A')}")

        # Format date
        decision_date = case.get('decision_date', '')
        if decision_date and len(decision_date) == 8:
            formatted_date = f"{decision_date[:4]}-{decision_date[4:6]}-{decision_date[6:]}"
        else:
            formatted_date = decision_date or 'N/A'
        output.append(f"Date: {formatted_date}")

        output.append(f"Tokens: {case.get('token_count', 0):,}")
        output.append(f"Relevance Statutes: {case.get('reference_statute', 'N/A')}")
        output.append("")
        output.append("=" * 32)
        output.append("JUDGMENT TEXT")
        output.append("=" * 32)
        output.append("")

        case_content = case.get('case_content', '[Content not available]')
        if case_content and len(case_content) > config.CASE_CONTENT_MAX_LENGTH:
            case_content = case_content[:config.CASE_CONTENT_MAX_LENGTH] + "\n\n... [truncated - full text exceeds limit]"
        output.append(case_content)

        return "\n".join(output)


class StatuteFormatter:
    """Format statute search results for MCP output"""

    @staticmethod
    def format_search_results(hits: List[Dict[str, Any]]) -> str:
        """
        Format statute search results in markdown.

        Returns formatted string matching the specification in function_design.md
        """
        if not hits:
            return "No statutes found matching your query."

        output = ["Available Statutes (top matches):\n"]
        output.append("Each result includes:")
        output.append("- statute_id: The unique identifier")
        output.append("- law_name: The name of the law in Korean")
        output.append("- abbreviation: Legal Abbreviation Name")
        output.append("- law_type: The type of law (법률, 대통령령, 대법원규칙 등)")
        output.append("- clause_count: Number of articles/clauses in the statute")
        output.append("- description: The full text of Article 1 (목적) of the statute")
        output.append("- citation_count: Number of times this statute has been cited by court cases")
        output.append("- token_count: Length of the full statute text in tokens\n")
        output.append("Choose statutes considering law name, type, citation count, description, "
                     "token count, and overall relevance to your use case.\n")

        for hit in hits:
            source = hit["_source"]
            score = hit["_score"]

            output.append("----------")
            output.append(f"- statute_id: {source.get('statute_id', 'N/A')}")
            output.append(f"- law_name: {source.get('law_name', 'N/A')}")
            output.append(f"- abbreviation: {source.get('abbreviation', 'N/A')}")
            output.append(f"- law_type: {source.get('law_type', 'N/A')}")
            output.append(f"- clause_count: {source.get('clause_count', 0)}")

            description = source.get('description', 'N/A')
            if description and len(description) > config.STATUTE_DESCRIPTION_MAX_LENGTH:
                description = description[:config.STATUTE_DESCRIPTION_MAX_LENGTH] + "..."
            output.append(f"- description: {description}")

            # Add citation count (판례 인용 횟수)
            citation_count = source.get('reference_case_count', 0)
            output.append(f"- citation_count: {citation_count}")

            output.append(f"- token_count: {source.get('token_count', 0):,}")

        output.append("----------")
        return "\n".join(output)

    @staticmethod
    def format_statute_content(
        statute_id: str,
        law_name: str,
        law_type: str,
        effective_date: str,
        promulgation_date: str,
        total_clauses: int,
        clauses: List[Dict[str, Any]],
        total_citation_count: int = 0,  # 법령 전체 인용 횟수
        abbreviation: str = "",
        law_detail_link: str = "",
        article_number: Optional[str] = None,
        article_range: Optional[str] = None
    ) -> str:
        """
        Format statute content in markdown.

        Returns formatted string matching the specification in function_design.md
        """
        output = ["=" * 16]
        output.append(f"STATUTE: {statute_id}")
        output.append("=" * 16)
        output.append("")
        output.append(f"Law Name: {law_name}")

        # Add abbreviation if available
        if abbreviation:
            output.append(f"Abbreviation: {abbreviation}")

        output.append(f"Law Type: {law_type}")

        # Format dates
        if effective_date and len(effective_date) == 8:
            effective_date = f"{effective_date[:4]}-{effective_date[4:6]}-{effective_date[6:]}"
        if promulgation_date and len(promulgation_date) == 8:
            promulgation_date = f"{promulgation_date[:4]}-{promulgation_date[4:6]}-{promulgation_date[6:]}"

        output.append(f"Effective Date: {effective_date}")
        output.append(f"Promulgation Date: {promulgation_date}")
        output.append(f"Total Clauses: {total_clauses}")
        output.append(f"Total Citation Count: {total_citation_count:,} cases")  # 법령 전체 인용 횟수

        # Show what was retrieved
        if article_number:
            output.append(f"Retrieved: {article_number}")
        elif article_range:
            output.append(f"Retrieved: {article_range}")
        else:
            output.append(f"Retrieved: Full statute ({len(clauses)} clauses)")

        output.append("")
        output.append("=" * 32)
        output.append("STATUTE TEXT")
        output.append("=" * 32)
        output.append("")

        # Add clause contents
        for clause in clauses:
            clause_num = clause.get('clause_number', '')
            clause_title = clause.get('clause_title', '')
            clause_content = clause.get('clause_content', '')
            clause_citation_count = clause.get('reference_case_count', 0)  # 조문별 인용 횟수

            # Format clause header
            if clause_title:
                clause_header = f"제{clause_num}조({clause_title})"
            else:
                clause_header = f"제{clause_num}조"

            # Always add citation count to header
            clause_header += f" [인용: {clause_citation_count:,}회]"

            output.append(clause_header)
            output.append(clause_content)
            output.append("")

        output.append("=" * 32)
        return "\n".join(output)

    @staticmethod
    def format_articles_list(
        statute_id: str,
        law_name: str,
        abbreviation: str,
        total_clauses: int,
        articles: List[Dict[str, Any]]
    ) -> str:
        """
        Format statute articles list (table of contents).

        Returns formatted string with article numbers, titles, and citation counts.
        """
        if not articles:
            return f"No articles found for statute '{statute_id}'."

        output = ["=" * 48]
        output.append(f"STATUTE ARTICLES LIST: {law_name}")
        if abbreviation:
            output.append(f"Abbreviation: {abbreviation}")
        output.append("=" * 48)
        output.append("")
        output.append(f"Statute ID: {statute_id}")
        output.append(f"Total Articles: {total_clauses:,}")
        output.append(f"Showing: All {len(articles)} articles")
        output.append("")
        output.append("-" * 48)
        output.append("TABLE OF CONTENTS")
        output.append("-" * 48)
        output.append("")

        # List all articles
        for article in articles:
            clause_num = article.get('clause_number', '')
            clause_title = article.get('clause_title', '')
            citation_count = article.get('reference_case_count', 0)

            # Format: 제750조(불법행위의 내용) [인용: 2,341회]
            if clause_title:
                line = f"제{clause_num}조({clause_title}) [인용: {citation_count:,}회]"
            else:
                line = f"제{clause_num}조 [인용: {citation_count:,}회]"

            output.append(line)

        output.append("")
        output.append("=" * 48)

        return "\n".join(output)


case_formatter = CaseFormatter()
statute_formatter = StatuteFormatter()
