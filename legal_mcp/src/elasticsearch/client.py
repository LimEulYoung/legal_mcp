"""Elasticsearch client wrapper"""
from typing import Optional, Dict, Any, List
from elasticsearch import Elasticsearch, AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError
import logging

from ..config import config

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """Elasticsearch client wrapper with connection management"""

    def __init__(self):
        self._client: Optional[AsyncElasticsearch] = None

    async def connect(self) -> AsyncElasticsearch:
        """Create and return async Elasticsearch client"""
        if self._client is None:
            auth = None
            if config.ES_USER and config.ES_PASSWORD:
                auth = (config.ES_USER, config.ES_PASSWORD)

            self._client = AsyncElasticsearch(
                [config.get_es_url()],
                basic_auth=auth,
                verify_certs=False,
                request_timeout=30
            )

            # Test connection
            try:
                await self._client.info()
                logger.info(f"Connected to Elasticsearch at {config.get_es_url()}")
            except ConnectionError as e:
                logger.error(f"Failed to connect to Elasticsearch: {e}")
                raise

        return self._client

    async def close(self):
        """Close Elasticsearch connection"""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Elasticsearch connection closed")

    async def search(
        self,
        index: str,
        query: Dict[str, Any],
        size: int = 10,
        source: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute search query"""
        client = await self.connect()

        try:
            response = await client.search(
                index=index,
                body=query,
                size=size,
                _source=source
            )
            return response
        except NotFoundError:
            logger.warning(f"Index {index} not found")
            return {"hits": {"hits": [], "total": {"value": 0}}}
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise

    async def get_document(
        self,
        index: str,
        doc_id: str,
        source: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        client = await self.connect()

        try:
            response = await client.get(
                index=index,
                id=doc_id,
                _source=source
            )
            return response["_source"]
        except NotFoundError:
            logger.warning(f"Document {doc_id} not found in index {index}")
            return None
        except Exception as e:
            logger.error(f"Get document error: {e}")
            raise

    async def search_by_field(
        self,
        index: str,
        field: str,
        value: Any,
        size: int = 1
    ) -> List[Dict[str, Any]]:
        """Search documents by exact field match"""
        query = {
            "query": {
                "term": {
                    field: value
                }
            }
        }

        response = await self.search(index, query, size)
        return [hit["_source"] for hit in response["hits"]["hits"]]

    async def get_statute_metadata(
        self,
        statute_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get statute metadata by statute ID.

        Returns metadata including:
        - clause_count: Total number of articles
        - reference_case_count: Total citation count across all cases
        - abbreviation: Legal abbreviation
        - law_name: Official law name
        """
        query = {
            "query": {
                "term": {"statute_id": statute_id}
            }
        }

        response = await self.search(
            index=config.INDEX_STATUTES_METADATA,
            query=query,
            size=1
        )

        hits = response["hits"]["hits"]
        if hits:
            return hits[0]["_source"]

        logger.warning(f"Statute metadata not found for ID: {statute_id}")
        return None


# Global client instance
es_client = ElasticsearchClient()
