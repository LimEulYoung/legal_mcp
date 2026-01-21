"""Embedding generation utility using Upstage API"""
import logging
from typing import List
from openai import AsyncOpenAI

from ..config import config

logger = logging.getLogger(__name__)


async def get_embedding(text: str) -> List[float]:
    """
    Generate embedding vector for text using Upstage Embedding API.

    Args:
        text: Text to embed

    Returns:
        List of floats representing the embedding vector (4096 dimensions)
    """
    try:
        logger.info(f"Requesting embedding for text: '{text[:50]}...'")

        client = AsyncOpenAI(
            api_key=config.UPSTAGE_API_KEY,
            base_url="https://api.upstage.ai/v1"
        )

        response = await client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=text
        )

        embedding = response.data[0].embedding
        logger.debug(f"Generated embedding (dim: {len(embedding)})")

        return embedding

    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise
