"""Reciprocal Rank Fusion algorithm for combining multiple ranking methods.

This module provides the ReciprocalRankFuser class that implements the
Reciprocal Rank Fusion (RRF) algorithm for combining rankings from multiple
retrieval systems. RRF is a simple yet effective method for fusing results
from different retrievers without requiring training data.

RRF uses the reciprocal rank of each document in individual rankings to
compute a fused score, providing a robust way to combine diverse retrieval
approaches.
"""

from typing import Any, Dict, List, Optional, Set
import logging

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class RRFResult(BaseModel):
    """Result from Reciprocal Rank Fusion.

    Attributes:
        id: Document ID.
        text: Document text content.
        rrf_score: Computed RRF score.
        rank: Final rank in fused results.
        source_ranks: Rank in each source ranking.
    """

    id: str = Field(..., description="Document ID")
    text: str = Field(..., description="Document text content")
    rrf_score: float = Field(..., description="RRF fusion score")
    rank: int = Field(..., description="Final rank in fused results")
    source_ranks: Dict[str, int] = Field(
        default_factory=dict, description="Rank in each source"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Document metadata"
    )


class ReciprocalRankFuser:
    """Reciprocal Rank Fusion for combining multiple retrieval rankings.

    Reciprocal Rank Fusion (RRF) is a simple and effective method for
    combining results from multiple retrieval systems. It uses the
    reciprocal of each document's rank in individual rankings to compute
    a fused score.

    The RRF score for a document d is computed as:
        RRF(d) = sum(1 / (k + rank_i(d)))
    where k is a constant (typically 60) and rank_i(d) is the rank of
    document d in ranking i.

    This algorithm has several advantages:
    - No training data required
    - Robust to missing documents in some rankings
    - Simple to implement and understand
    - Good empirical performance

    Example:
        >>> fuser = ReciprocalRankFuser()
        >>> rankings = [
        ...     [{"id": "1", "text": "doc1"}, {"id": "2", "text": "doc2"}],
        ...     [{"id": "2", "text": "doc2"}, {"id": "3", "text": "doc3"}],
        ... ]
        >>> results = fuser.fuse(rankings)
    """

    def __init__(
        self, k: int = 60, max_rank: int = 1000, normalize: bool = True
    ) -> None:
        """Initialize the RRF fuser.

        Args:
            k: Ranking constant for RRF formula. Higher values reduce
                the impact of lower ranks. Default of 60 is recommended
                based on empirical evaluation in the original paper.
            max_rank: Maximum rank to consider. Documents ranked beyond
                this are excluded from fusion.
            normalize: Whether to normalize scores to [0, 1] range.
        """
        if k < 1:
            raise ValueError("k must be positive")

        if max_rank < 1:
            raise ValueError("max_rank must be positive")

        self._k = k
        self._max_rank = max_rank
        self._normalize = normalize

    def fuse(
        self,
        rankings: List[List[Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[RRFResult]:
        """Fuse multiple rankings using Reciprocal Rank Fusion.

        Takes a list of rankings (each ranking is a list of documents ordered
        by relevance) and returns a single fused ranking.

        Args:
            rankings: List of rankings, where each ranking is a list of
                documents with at least 'id' and 'text' fields.
            top_k: Number of top results to return. If None, returns all.

        Returns:
            List of RRFResult objects sorted by RRF score.

        Raises:
            ValueError: If rankings is empty or contains empty rankings.
        """
        if not rankings:
            raise ValueError("rankings cannot be empty")

        if any(not r for r in rankings):
            raise ValueError("Cannot fuse with empty rankings")

        doc_scores: Dict[str, Dict[str, Any]] = {}

        for rank_idx, ranking in enumerate(rankings):
            for rank, doc in enumerate(ranking, start=1):
                if rank > self._max_rank:
                    break

                doc_id = doc.get("id")
                if not doc_id:
                    logger.warning(f"Skipping document without ID at rank {rank}")
                    continue

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "text": doc.get("text", ""),
                        "rrf_score": 0.0,
                        "source_ranks": {},
                        "metadata": doc.get("metadata"),
                    }

                source_rank = doc.get("source_rank", rank)
                rrf_contribution = 1.0 / (self._k + source_rank)
                doc_scores[doc_id]["rrf_score"] += rrf_contribution
                doc_scores[doc_id]["source_ranks"][f"ranker_{rank_idx}"] = source_rank

        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1]["rrf_score"],
            reverse=True,
        )

        if self._normalize and sorted_docs:
            max_score = sorted_docs[0][1]["rrf_score"]
            if max_score > 0:
                for doc_id, scores in sorted_docs:
                    scores["rrf_score"] = scores["rrf_score"] / max_score

        results: List[RRFResult] = []
        for rank, (doc_id, scores) in enumerate(sorted_docs, start=1):
            if top_k and rank > top_k:
                break

            results.append(
                RRFResult(
                    id=doc_id,
                    text=scores["text"],
                    rrf_score=scores["rrf_score"],
                    rank=rank,
                    source_ranks=scores["source_ranks"],
                    metadata=scores.get("metadata"),
                )
            )

        return results

    def fuse_with_sources(
        self,
        source_rankings: Dict[str, List[Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[RRFResult]:
        """Fuse rankings with named sources.

        Similar to fuse() but uses source names as keys instead of indices.

        Args:
            source_rankings: Dictionary mapping source names to rankings.
            top_k: Number of top results to return.

        Returns:
            List of RRFResult objects with named source ranks.

        Raises:
            ValueError: If source_rankings is empty.
        """
        if not source_rankings:
            raise ValueError("source_rankings cannot be empty")

        rankings = list(source_rankings.values())
        source_names = list(source_rankings.keys())

        doc_scores: Dict[str, Dict[str, Any]] = {}

        for source_name, ranking in source_rankings.items():
            for rank, doc in enumerate(ranking, start=1):
                if rank > self._max_rank:
                    break

                doc_id = doc.get("id")
                if not doc_id:
                    continue

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "text": doc.get("text", ""),
                        "rrf_score": 0.0,
                        "source_ranks": {},
                        "metadata": doc.get("metadata"),
                    }

                rrf_contribution = 1.0 / (self._k + rank)
                doc_scores[doc_id]["rrf_score"] += rrf_contribution
                doc_scores[doc_id]["source_ranks"][source_name] = rank

        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1]["rrf_score"],
            reverse=True,
        )

        if self._normalize and sorted_docs:
            max_score = sorted_docs[0][1]["rrf_score"]
            if max_score > 0:
                for doc_id, scores in sorted_docs:
                    scores["rrf_score"] = scores["rrf_score"] / max_score

        results: List[RRFResult] = []
        for rank, (doc_id, scores) in enumerate(sorted_docs, start=1):
            if top_k and rank > top_k:
                break

            results.append(
                RRFResult(
                    id=doc_id,
                    text=scores["text"],
                    rrf_score=scores["rrf_score"],
                    rank=rank,
                    source_ranks=scores["source_ranks"],
                    metadata=scores.get("metadata"),
                )
            )

        return results

    def interpolate(
        self,
        rankings: List[List[Dict[str, Any]]],
        weights: Optional[List[float]] = None,
        top_k: Optional[int] = None,
    ) -> List[RRFResult]:
        """Fuse rankings with interpolation weights.

        Combines RRF with linear interpolation of scores from each ranking.

        Args:
            rankings: List of rankings.
            weights: Optional weights for each ranking. Must sum to 1.0.
                If None, equal weights are used.
            top_k: Number of top results to return.

        Returns:
            List of interpolated RRFResult objects.

        Raises:
            ValueError: If weights don't sum to 1.0 or have wrong length.
        """
        n_rankings = len(rankings)

        if weights is not None:
            if len(weights) != n_rankings:
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of rankings ({n_rankings})"
                )
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
        else:
            weights = [1.0 / n_rankings] * n_rankings

        doc_info: Dict[str, Dict[str, Any]] = {}

        for rank_idx, ranking in enumerate(rankings):
            weight = weights[rank_idx]
            for rank, doc in enumerate(ranking, start=1):
                if rank > self._max_rank:
                    break

                doc_id = doc.get("id")
                if not doc_id:
                    continue

                if doc_id not in doc_info:
                    doc_info[doc_id] = {
                        "text": doc.get("text", ""),
                        "interpolated_score": 0.0,
                        "source_ranks": {},
                        "rrf_score": 0.0,
                        "metadata": doc.get("metadata"),
                    }

                rrf_score = 1.0 / (self._k + rank)
                doc_info[doc_id]["rrf_score"] += rrf_score

                if "score" in doc:
                    weighted_score = doc["score"] * weight
                    doc_info[doc_id]["interpolated_score"] += weighted_score

                doc_info[doc_id]["source_ranks"][f"ranker_{rank_idx}"] = rank

        sorted_docs = sorted(
            doc_info.items(),
            key=lambda x: x[1]["rrf_score"],
            reverse=True,
        )

        results: List[RRFResult] = []
        for rank, (doc_id, info) in enumerate(sorted_docs, start=1):
            if top_k and rank > top_k:
                break

            results.append(
                RRFResult(
                    id=doc_id,
                    text=info["text"],
                    rrf_score=info["rrf_score"],
                    rank=rank,
                    source_ranks=info["source_ranks"],
                    metadata=info.get("metadata"),
                )
            )

        return results

    @property
    def k(self) -> int:
        """Get the RRF k parameter."""
        return self._k

    @property
    def max_rank(self) -> int:
        """Get the maximum rank parameter."""
        return self._max_rank

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ReciprocalRankFuser(k={self._k}, max_rank={self._max_rank})"
