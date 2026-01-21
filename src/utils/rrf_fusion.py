"""
RRF (Reciprocal Rank Fusion) implementation
- Fuses BM25 and vector search results using rank-based scoring
- Solves scale difference between BM25 (10-50) and vector (0-1) scores
"""

from typing import Dict, List
from collections import defaultdict


class RRFFusion:
    """RRF (Reciprocal Rank Fusion) implementation"""

    @staticmethod
    def compute_rrf_score(rank: int, k: int = 60) -> float:
        """
        Compute RRF score for a given rank.

        Args:
            rank: Rank position (0-indexed)
            k: RRF constant (default: 60)

        Returns:
            RRF score
        """
        return 1.0 / (rank + k)

    @staticmethod
    def fuse_elasticsearch_hits(
        bm25_hits: List[Dict],
        vector_hits: List[Dict],
        k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0
    ) -> List[Dict]:
        """
        Fuse BM25 and vector search results using RRF.

        Args:
            bm25_hits: BM25 search hits (ES response['hits']['hits'])
            vector_hits: Vector search hits
            k: RRF constant
            bm25_weight: BM25 result weight
            vector_weight: Vector result weight

        Returns:
            RRF-fused hits sorted by RRF score
        """
        # Store RRF scores and metadata per document
        doc_scores = defaultdict(lambda: {
            'rrf_score': 0.0,
            'bm25_rank': None,
            'vector_rank': None,
            'bm25_score': None,
            'vector_score': None,
            'hit': None
        })

        # Process BM25 results
        for rank, hit in enumerate(bm25_hits):
            doc_id = hit['_id']
            rrf_contribution = RRFFusion.compute_rrf_score(rank, k) * bm25_weight
            doc_scores[doc_id]['rrf_score'] += rrf_contribution
            doc_scores[doc_id]['bm25_rank'] = rank
            doc_scores[doc_id]['bm25_score'] = hit.get('_score', 0)
            doc_scores[doc_id]['hit'] = hit

        # Process vector results
        for rank, hit in enumerate(vector_hits):
            doc_id = hit['_id']
            rrf_contribution = RRFFusion.compute_rrf_score(rank, k) * vector_weight
            doc_scores[doc_id]['rrf_score'] += rrf_contribution
            doc_scores[doc_id]['vector_rank'] = rank
            doc_scores[doc_id]['vector_score'] = hit.get('_score', 0)
            # Save hit if not already present (vector-only results)
            if doc_scores[doc_id]['hit'] is None:
                doc_scores[doc_id]['hit'] = hit

        # Sort by RRF score
        fused_hits = []
        for doc_id, data in sorted(
            doc_scores.items(),
            key=lambda x: x[1]['rrf_score'],
            reverse=True
        ):
            hit = data['hit'].copy()
            # Add RRF metadata to _source
            if '_source' not in hit:
                hit['_source'] = {}

            hit['_source']['rrf_score'] = data['rrf_score']
            hit['_source']['bm25_rank'] = data['bm25_rank']
            hit['_source']['vector_rank'] = data['vector_rank']
            hit['_source']['bm25_original_score'] = data['bm25_score']
            hit['_source']['vector_original_score'] = data['vector_score']

            # Replace _score with rrf_score
            hit['_score'] = data['rrf_score']

            fused_hits.append(hit)

        return fused_hits


rrf_fusion = RRFFusion()
