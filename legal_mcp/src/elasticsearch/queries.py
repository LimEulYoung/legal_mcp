"""Elasticsearch query builders"""
from typing import Dict, Any, List, Optional
import logging

from ..config import config

logger = logging.getLogger(__name__)


class QueryBuilder:
    """Build Elasticsearch queries for RRF-based hybrid search"""

    @staticmethod
    def build_statute_content_query(
        statute_id: str,
        article_number: Optional[str] = None,
        article_range: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build query to retrieve statute content.

        Args:
            statute_id: Statute identifier
            article_number: Specific article (e.g., "15")
            article_range: Article range (e.g., "1-10")
        """
        must_queries = [
            {"term": {"statute_id": statute_id}}
        ]

        if article_number:
            must_queries.append({
                "term": {"clause_number": article_number}
            })

        elif article_range:
            # Parse range like "1-10"
            try:
                start, end = article_range.split("-")
                start_num = int(start.strip())
                end_num = int(end.strip())

                # Use script query to convert clause_number to int and compare
                # Handles main clause numbers (ignores sub-clauses like "11-2")
                must_queries.append({
                    "script": {
                        "script": {
                            "source": """
                                String clauseNum = doc['clause_number'].value;
                                int mainNum;
                                try {
                                    // Extract number before hyphen (if exists)
                                    int hyphenIndex = clauseNum.indexOf('-');
                                    if (hyphenIndex > 0) {
                                        mainNum = Integer.parseInt(clauseNum.substring(0, hyphenIndex));
                                    } else {
                                        mainNum = Integer.parseInt(clauseNum);
                                    }
                                    return mainNum >= params.start && mainNum <= params.end;
                                } catch (Exception e) {
                                    return false;
                                }
                            """,
                            "params": {
                                "start": start_num,
                                "end": end_num
                            }
                        }
                    }
                })
            except ValueError as e:
                logger.error(f"Invalid article_range format: {article_range}")
                raise ValueError(
                    f"Invalid article_range format: '{article_range}'. "
                    f"Expected format: '1-10' (hyphen-separated range)"
                )

        query = {
            "query": {
                "bool": {
                    "must": must_queries
                }
            },
            "sort": [
                {"clause_number": "asc"}
            ]
        }

        logger.debug(f"Built statute content query: {query}")
        return query

    @staticmethod
    def build_statute_articles_list_query(statute_id: str) -> Dict[str, Any]:
        """
        Build query to retrieve statute articles list (titles only).

        Args:
            statute_id: Statute identifier
        """
        query = {
            "query": {
                "term": {"statute_id": statute_id}
            },
            "_source": ["clause_number", "clause_title", "reference_case_count", "law_name"],
            "sort": [
                {"clause_number": "asc"}
            ]
        }

        logger.debug(f"Built statute articles list query: {query}")
        return query


    @staticmethod
    def build_bm25_only_case_query(
        query: str,
        reference_statute: str = None,
        court_name: str = None,
        date_from: str = None,
        date_to: str = None
    ) -> Dict[str, Any]:
        """
        Build BM25-only search query for court cases (for RRF).

        Args:
            query: Main search text
            reference_statute: Statute reference filter
            court_name: Court name filter
            date_from: Start date in YYYYMMDD format
            date_to: End date in YYYYMMDD format

        Returns:
            BM25-only Elasticsearch query
        """
        # BM25 query
        must_queries = []

        if query:
            must_queries.append({
                "multi_match": {
                    "query": query,
                    "fields": [
                        f"case_number^{config.CASE_NUMBER_BOOST}",
                        f"case_name^{config.CASE_NAME_BOOST}",
                        f"reference_statute^{config.CASE_REFERENCE_STATUTE_BOOST}",
                        f"judged_statute^{config.CASE_JUDGED_STATUTE_BOOST}",
                        f"judgment_summary^{config.CASE_JUDGMENT_SUMMARY_BOOST}",
                        f"case_content^{config.CASE_CONTENT_BOOST}"
                    ],
                    "type": "best_fields"
                }
            })

        # Filters
        filter_queries = []

        if reference_statute:
            statute_pattern = f"*{reference_statute}*"
            filter_queries.append({
                "bool": {
                    "should": [
                        {"wildcard": {"reference_statute.keyword": statute_pattern}},
                        {"wildcard": {"judged_statute.keyword": statute_pattern}}
                    ],
                    "minimum_should_match": 1
                }
            })

        if court_name:
            filter_queries.append({"term": {"court_name": court_name}})

        if date_from or date_to:
            date_range = {}
            if date_from:
                date_range["gte"] = date_from
            if date_to:
                date_range["lte"] = date_to
            filter_queries.append({"range": {"decision_date": date_range}})

        # Function score boosting
        functions = [
            {
                "field_value_factor": {
                    "field": "court_level_score",
                    "factor": config.CASE_COURT_LEVEL_FACTOR,
                    "missing": 0
                }
            },
            {
                "script_score": {
                    "script": {
                        "source": """
                            if (doc['decision_date'].size() == 0) {
                                return params.missing_default;
                            }
                            long now = new Date().getTime();
                            long docDate = doc['decision_date'].value.toInstant().toEpochMilli();
                            long diff = now - docDate;
                            double years = diff / (365.25 * 24 * 60 * 60 * 1000.0);
                            double scale = params.scale_years;
                            double decay = params.decay;
                            double max_score = params.max_score;
                            if (years <= 0) return max_score;
                            if (years >= scale) return max_score * decay;
                            double factor = 1.0 - (years / scale) * (1.0 - decay);
                            return max_score * factor;
                        """,
                        "params": {
                            "max_score": config.CASE_RECENCY_MAX_SCORE,
                            "decay": config.CASE_RECENCY_DECAY,
                            "scale_years": config.CASE_RECENCY_SCALE_YEARS,
                            "missing_default": config.CASE_RECENCY_MISSING_DEFAULT
                        }
                    }
                }
            },
            {
                "field_value_factor": {
                    "field": "reference_case_count",
                    "factor": config.CASE_CITATION_FACTOR,
                    "missing": 0,
                    "modifier": "log1p"
                }
            }
        ]

        return {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": must_queries if must_queries else [{"match_all": {}}],
                            "filter": filter_queries
                        }
                    },
                    "functions": functions,
                    "score_mode": "sum",
                    "boost_mode": "sum",
                    "max_boost": config.CASE_MAX_BOOST
                }
            },
            "_source": {
                "excludes": ["case_content", "embedding_vector"]
            }
        }

    @staticmethod
    def build_vector_only_case_query(
        embedding: List[float],
        reference_statute: str = None,
        court_name: str = None,
        date_from: str = None,
        date_to: str = None
    ) -> Dict[str, Any]:
        """
        Build vector-only search query for court cases (for RRF).

        Args:
            embedding: Query embedding vector
            reference_statute: Statute reference filter
            court_name: Court name filter
            date_from: Start date in YYYYMMDD format
            date_to: End date in YYYYMMDD format

        Returns:
            Vector-only Elasticsearch query
        """
        # Filters
        filter_queries = []

        if reference_statute:
            statute_pattern = f"*{reference_statute}*"
            filter_queries.append({
                "bool": {
                    "should": [
                        {"wildcard": {"reference_statute.keyword": statute_pattern}},
                        {"wildcard": {"judged_statute.keyword": statute_pattern}}
                    ],
                    "minimum_should_match": 1
                }
            })

        if court_name:
            filter_queries.append({"term": {"court_name": court_name}})

        if date_from or date_to:
            date_range = {}
            if date_from:
                date_range["gte"] = date_from
            if date_to:
                date_range["lte"] = date_to
            filter_queries.append({"range": {"decision_date": date_range}})

        # KNN query
        knn_query = {
            "field": "embedding_vector",
            "query_vector": embedding,
            "k": config.CASE_KNN_K,
            "num_candidates": config.CASE_KNN_NUM_CANDIDATES
        }

        # Add filters if present
        if filter_queries:
            knn_query["filter"] = filter_queries

        return {
            "knn": knn_query,
            "_source": {
                "excludes": ["case_content", "embedding_vector"]
            }
        }

    @staticmethod
    def build_bm25_only_statute_query(
        query: str,
        law_type: str = None
    ) -> Dict[str, Any]:
        """
        Build BM25-only search query for statutes (for RRF).

        Args:
            query: Main search text
            law_type: Law type filter

        Returns:
            BM25-only Elasticsearch query
        """
        must_queries = []

        if query:
            must_queries.append({
                "multi_match": {
                    "query": query,
                    "fields": [
                        f"law_name^{config.STATUTE_LAW_NAME_BOOST}",
                        f"abbreviation^{config.STATUTE_ABBREVIATION_BOOST}",
                        f"description^{config.STATUTE_DESCRIPTION_BOOST}"
                    ],
                    "type": "best_fields"
                }
            })

        # Filters
        filter_queries = []

        if law_type:
            filter_queries.append({"term": {"law_type": law_type}})

        # Boosting
        functions = [
            {
                "field_value_factor": {
                    "field": "reference_case_count",
                    "factor": config.STATUTE_CITATION_FACTOR,
                    "missing": 0,
                    "modifier": "log1p"
                }
            }
        ]

        return {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "should": must_queries if must_queries else [],
                            "filter": filter_queries
                        }
                    },
                    "functions": functions,
                    "score_mode": "sum",
                    "boost_mode": "sum",
                    "max_boost": config.STATUTE_MAX_BOOST
                }
            },
            "_source": {
                "excludes": ["embedding_vector"]
            }
        }

    @staticmethod
    def build_vector_only_statute_query(
        embedding: List[float],
        law_type: str = None
    ) -> Dict[str, Any]:
        """
        Build vector-only search query for statutes (for RRF).

        Args:
            embedding: Query embedding vector
            law_type: Law type filter

        Returns:
            Vector-only Elasticsearch query
        """
        # Filters
        filter_queries = []

        if law_type:
            filter_queries.append({"term": {"law_type": law_type}})

        # KNN query
        knn_query = {
            "field": "embedding_vector",
            "query_vector": embedding,
            "k": config.STATUTE_KNN_K,
            "num_candidates": config.STATUTE_KNN_NUM_CANDIDATES
        }

        # Add filters if present
        if filter_queries:
            knn_query["filter"] = filter_queries

        return {
            "knn": knn_query,
            "_source": {
                "excludes": ["embedding_vector"]
            }
        }


query_builder = QueryBuilder()
