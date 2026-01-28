"""Complete reranking pipeline combining hybrid search + cross-encoder.

This module provides the RerankingPipeline class that orchestrates the
complete search pipeline: hybrid search (vector + BM25) followed by
cross-encoder reranking for improved result quality.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

from pydantic import BaseModel, Field

from knowledge_base.storage.hybrid_search import HybridSearchEngine, HybridSearchResult
from knowledge_base.reranking.cross_encoder import (
    CrossEncoderReranker,
    RerankedSearchResult,
)
from knowledge_base.reranking.rrf_fuser import ReciprocalRankFuser, RRFResult


logger = logging.getLogger(__name__)


class RerankedSearchResultWithExplanation(BaseModel):
    """Reranked result with detailed explanation of ranking decisions.

    Attributes:
        id: Document/chunk ID.
        text: Text content.
        vector_score: Vector similarity score from hybrid search.
        bm25_score: BM25 score from hybrid search.
        final_score: Combined hybrid search score.
        cross_encoder_score: Cross-encoder relevance score.
        reranked_score: Final reranked score.
        metadata: Document metadata.
        source: Result source from hybrid search.
        explanation: Human-readable explanation of ranking decision.
        rank_factors: Breakdown of factors contributing to ranking.
    """

    id: str = Field(..., description="Document or chunk ID")
    text: str = Field(..., description="Text content")
    vector_score: float = Field(default=0.0, description="Vector similarity score")
    bm25_score: float = Field(default=0.0, description="BM25 score")
    final_score: float = Field(default=0.0, description="Combined hybrid score")
    cross_encoder_score: float = Field(
        default=0.0, description="Cross-encoder relevance score"
    )
    reranked_score: float = Field(default=0.0, description="Final reranked score")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Document metadata"
    )
    source: str = Field(default="hybrid", description="Result source")
    explanation: str = Field(default="", description="Explanation of ranking decision")
    rank_factors: Dict[str, float] = Field(
        default_factory=dict, description="Breakdown of ranking factors"
    )


class RerankingPipeline:
    """Complete reranking pipeline combining hybrid search + cross-encoder.

    This pipeline orchestrates the full search workflow:
    1. Execute hybrid search (vector + BM25) to get initial candidates
    2. Optionally generate multiple query variations
    3. Rerank candidates using cross-encoder for improved relevance
    4. Optionally fuse results using Reciprocal Rank Fusion

    The pipeline provides a unified interface for high-quality search
    results suitable for production use cases.

    Example:
        >>> pipeline = RerankingPipeline(hybrid_engine, reranker)
        >>> results = await pipeline.search(
        ...     query="machine learning techniques",
        ...     initial_top_k=50,
        ...     final_top_k=10,
        ... )
    """

    def __init__(
        self,
        hybrid_search: HybridSearchEngine,
        cross_encoder: CrossEncoderReranker,
        rr_fuser: Optional[ReciprocalRankFuser] = None,
    ) -> None:
        """Initialize the reranking pipeline.

        Args:
            hybrid_search: HybridSearchEngine instance for initial retrieval.
            cross_encoder: CrossEncoderReranker instance for reranking.
            rr_fuser: Optional ReciprocalRankFuser for multi-query fusion.
        """
        if hybrid_search is None:
            raise ValueError("hybrid_search cannot be None")

        if cross_encoder is None:
            raise ValueError("cross_encoder cannot be None")

        self._hybrid = hybrid_search
        self._cross_encoder = cross_encoder
        self._rr_fuser = rr_fuser or ReciprocalRankFuser()

    async def search(
        self,
        query: str,
        initial_top_k: int = 50,
        final_top_k: int = 10,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        filters: Optional[Dict] = None,
        domain: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[RerankedSearchResult]:
        """Execute complete reranking pipeline.

        Performs hybrid search to get candidates, then reranks using
        cross-encoder for improved relevance scoring.

        Args:
            query: Search query string.
            initial_top_k: Number of candidates to retrieve from hybrid search.
            final_top_k: Number of results to return after reranking.
            vector_weight: Weight for vector similarity in fusion.
            bm25_weight: Weight for BM25 in fusion.
            filters: Optional metadata filters.
            domain: Optional domain filter.
            query_embedding: Pre-computed query embedding.

        Returns:
            List of RerankedSearchResult objects sorted by reranked score.

        Raises:
            ValueError: If parameters are invalid.
            RuntimeError: If pipeline components are not initialized.
        """
        if initial_top_k < final_top_k:
            raise ValueError("initial_top_k must be >= final_top_k")

        if abs(vector_weight + bm25_weight - 1.0) > 1e-6:
            raise ValueError("vector_weight and bm25_weight must sum to 1.0")

        if final_top_k < 1:
            raise ValueError("final_top_k must be positive")

        logger.info(
            f"Executing reranking pipeline: query='{query[:50]}...', "
            f"initial_k={initial_top_k}, final_k={final_top_k}"
        )

        candidates = await self._hybrid.search(
            query=query,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            top_k=initial_top_k,
            filters=filters,
            domain=domain,
            query_embedding=query_embedding,
        )

        if not candidates:
            logger.info("No candidates found from hybrid search")
            return []

        logger.info(
            f"Retrieved {len(candidates)} candidates, reranking with cross-encoder"
        )

        results = await self._cross_encoder.rerank(
            query=query,
            candidates=candidates,
            top_k=final_top_k,
        )

        logger.info(f"Returning {len(results)} reranked results")
        return results

    async def search_with_explanation(
        self,
        query: str,
        initial_top_k: int = 50,
        final_top_k: int = 10,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        filters: Optional[Dict] = None,
        domain: Optional[str] = None,
    ) -> List[RerankedSearchResultWithExplanation]:
        """Search with detailed explanation of reranking decisions.

        Returns results with human-readable explanations of why each
        document was ranked as it was, including breakdown of scoring factors.

        Args:
            query: Search query string.
            initial_top_k: Number of candidates to retrieve.
            final_top_k: Number of results to return.
            vector_weight: Weight for vector similarity.
            bm25_weight: Weight for BM25.
            filters: Optional metadata filters.
            domain: Optional domain filter.

        Returns:
            List of RerankedSearchResultWithExplanation objects.
        """
        if not self._cross_encoder.is_initialized:
            await self._cross_encoder.initialize()

        candidates = await self._hybrid.search(
            query=query,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            top_k=initial_top_k,
            filters=filters,
            domain=domain,
        )

        if not candidates:
            return []

        results, ce_scores = await self._cross_encoder.rerank_with_scores(
            query=query,
            candidates=candidates,
            top_k=final_top_k,
        )

        explained_results: List[RerankedSearchResultWithExplanation] = []

        for result, candidate in zip(results, candidates):
            ce_score = result.cross_encoder_score
            hybrid_score = candidate.final_score

            factors = {
                "vector_score": candidate.vector_score,
                "bm25_score": candidate.bm25_score,
                "hybrid_fusion": hybrid_score,
                "cross_encoder": ce_score,
                "rerank_weight": 0.7,
            }

            explanation = self._generate_explanation(
                query=query,
                candidate=candidate,
                ce_score=ce_score,
                hybrid_score=hybrid_score,
            )

            explained_results.append(
                RerankedSearchResultWithExplanation(
                    id=result.id,
                    text=result.text,
                    vector_score=candidate.vector_score,
                    bm25_score=candidate.bm25_score,
                    final_score=hybrid_score,
                    cross_encoder_score=ce_score,
                    reranked_score=result.reranked_score,
                    metadata=result.metadata,
                    source=result.source,
                    explanation=explanation,
                    rank_factors=factors,
                )
            )

        return explained_results

    def _generate_explanation(
        self,
        query: str,
        candidate: HybridSearchResult,
        ce_score: float,
        hybrid_score: float,
    ) -> str:
        """Generate human-readable explanation for ranking.

        Args:
            query: Original search query.
            candidate: Original hybrid search result.
            ce_score: Cross-encoder score.
            hybrid_score: Combined hybrid score.

        Returns:
            Human-readable explanation string.
        """
        parts = []

        if candidate.vector_score > 0.5:
            parts.append("strong semantic match (high vector score)")

        if candidate.bm25_score > 0.5:
            parts.append("relevant keyword match (BM25)")

        if ce_score > 0.5:
            parts.append("high cross-encoder relevance")
        elif ce_score < 0.1:
            parts.append("low cross-encoder relevance")

        if candidate.source == "hybrid":
            parts.append("found by both vector and keyword search")

        if not parts:
            parts.append("moderate relevance across methods")

        explanation = f"Document ranked highly due to: {', '.join(parts)}."
        explanation += f" Cross-encoder relevance: {ce_score:.3f}."

        return explanation

    async def search_with_rrf(
        self,
        queries: List[str],
        initial_top_k: int = 50,
        final_top_k: int = 10,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        filters: Optional[Dict] = None,
        domain: Optional[str] = None,
    ) -> List[RRFResult]:
        """Search with multiple queries and RRF fusion.

        Executes the pipeline for multiple query variations and fuses
        results using Reciprocal Rank Fusion.

        Args:
            queries: List of query variations to search.
            initial_top_k: Candidates per query.
            final_top_k: Final results after fusion.
            vector_weight: Weight for vector similarity.
            bm25_weight: Weight for BM25.
            filters: Optional metadata filters.
            domain: Optional domain filter.

        Returns:
            List of RRFResult objects from fusion.
        """
        if not queries:
            raise ValueError("queries cannot be empty")

        if not self._cross_encoder.is_initialized:
            await self._cross_encoder.initialize()

        rankings: List[List[Dict[str, Any]]] = []

        for query in queries:
            candidates = await self._hybrid.search(
                query=query,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                top_k=initial_top_k,
                filters=filters,
                domain=domain,
            )

            if not candidates:
                continue

            reranked = await self._cross_encoder.rerank(
                query=query,
                candidates=candidates,
                top_k=initial_top_k,
            )

            ranking = [
                {
                    "id": r.id,
                    "text": r.text,
                    "score": r.reranked_score,
                    "source_rank": idx + 1,
                    "metadata": r.metadata,
                }
                for idx, r in enumerate(reranked)
            ]

            rankings.append(ranking)

        if not rankings:
            return []

        fused = self._rr_fuser.fuse(rankings, top_k=final_top_k)
        return fused

    async def health_check(self) -> Dict[str, Any]:
        """Check health of pipeline components.

        Returns:
            Dictionary with health status of each component.
        """
        ce_healthy = self._cross_encoder.is_initialized

        return {
            "cross_encoder_healthy": ce_healthy,
            "cross_encoder_model": self._cross_encoder.model_name
            if ce_healthy
            else None,
            "rr_fuser_config": {
                "k": self._rr_fuser.k,
                "max_rank": self._rr_fuser.max_rank,
            },
            "overall_healthy": ce_healthy,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RerankingPipeline("
            f"cross_encoder='{self._cross_encoder.model_name}', "
            f"rr_fuser={self._rr_fuser}"
        )
