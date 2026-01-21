"""Configuration for Legal Search MCP Server"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)


class Config:
    """Application configuration"""

    # Elasticsearch settings
    ES_HOST: str = os.getenv("ES_HOST")
    ES_PORT: int = int(os.getenv("ES_PORT", "9200"))
    ES_SCHEME: str = os.getenv("ES_SCHEME", "http")
    ES_USER: Optional[str] = os.getenv("ES_USER")
    ES_PASSWORD: Optional[str] = os.getenv("ES_PASSWORD")

    # Index names
    INDEX_COURT_CASES: str = "test_court_cases_new"
    INDEX_STATUTES_METADATA: str = "test_statutes_metadata"
    INDEX_STATUTES: str = "test_statutes"

    # Upstage API settings
    UPSTAGE_API_KEY: str = os.getenv("UPSTAGE_API_KEY", "")

    # Embedding settings
    EMBEDDING_MODEL: str = "embedding-query"
    EMBEDDING_DIMENSIONS: int = 4096

    # Search settings
    SEARCH_TOP_K: int = 5  # Fixed number of search results to return
    CASE_SUMMARY_MAX_LENGTH: int = 1500  # Max characters for judgment_summary in search results
    STATUTE_DESCRIPTION_MAX_LENGTH: int = 1000  # Max characters for statute description (Article 1) in search results
    CASE_CONTENT_MAX_LENGTH: int = 5000  # Max characters for full case content output

    # === RRF (Reciprocal Rank Fusion) Settings ===
    RRF_K: int = 60  # RRF constant (typically 60)
    RRF_BM25_WEIGHT: float = 1.05  # BM25 result weight in RRF
    RRF_VECTOR_WEIGHT: float = 1.0  # Vector result weight in RRF

    # Court level boosting scores (used in Elasticsearch field mapping)
    COURT_LEVEL_SCORES: dict = {
        "대법원": 10,
        "헌법재판소": 10,
        "고등법원": 7,
        "default": 5
    }

    # === Case Search Boosting Parameters ===

    # Field boost weights (BM25 multi-match)
    CASE_NUMBER_BOOST: float = 1.5
    CASE_NAME_BOOST: float = 0.8
    CASE_REFERENCE_STATUTE_BOOST: float = 3.5
    CASE_JUDGED_STATUTE_BOOST: float = 3.5
    CASE_JUDGMENT_SUMMARY_BOOST: float = 6.0
    CASE_CONTENT_BOOST: float = 2.5

    # Function score boosting
    CASE_COURT_LEVEL_FACTOR: float = 0.0  # Max 4 points (10 * 0.4)
    CASE_RECENCY_MAX_SCORE: float = 0.0  # Max points for recency
    CASE_RECENCY_DECAY: float = 0.8  # Decay rate for old cases
    CASE_RECENCY_SCALE_YEARS: float = 50.0  # Years for full decay
    CASE_RECENCY_MISSING_DEFAULT: float = 0.0  # Default for missing dates (assumes 2010)
    CASE_CITATION_FACTOR: float = 0.0  # Citation count factor (log1p(count) * factor)
    CASE_MAX_BOOST: float = 0.0  # Total max boost points

    # KNN parameters
    CASE_KNN_K: int = 30  # Number of nearest neighbors
    CASE_KNN_NUM_CANDIDATES: int = 150  # Number of candidates to consider

    # === Statute Search Boosting Parameters ===

    # Field boost weights (BM25 multi-match)
    STATUTE_LAW_NAME_BOOST: float = 3.0
    STATUTE_ABBREVIATION_BOOST: float = 2.0
    STATUTE_DESCRIPTION_BOOST: float = 1.0

    # Function score boosting
    STATUTE_CITATION_FACTOR: float = 0.0  # Citation boost factor (log1p)
    STATUTE_MAX_BOOST: float = 0.0  # Max citation boost points

    # KNN parameters
    STATUTE_KNN_K: int = 30  # Number of nearest neighbors
    STATUTE_KNN_NUM_CANDIDATES: int = 150  # Number of candidates to consider

    @classmethod
    def get_es_url(cls) -> str:
        """Get Elasticsearch URL"""
        return f"{cls.ES_SCHEME}://{cls.ES_HOST}:{cls.ES_PORT}"

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration"""
        if not cls.ES_HOST:
            raise ValueError("ES_HOST is required in .env file")
        if not cls.UPSTAGE_API_KEY:
            raise ValueError("UPSTAGE_API_KEY is required in .env file")


config = Config()
