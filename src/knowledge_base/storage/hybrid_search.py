"""Hybrid search engine combining vector similarity and BM25 keyword search.

This module provides a unified search interface that combines semantic similarity
search (using vector embeddings) with keyword-based BM25 search. The results
from both search methods are fused using weighted score normalization and
combination to provide optimal search results.

The fusion algorithm uses:
1. Min-max normalization to scale both vector and BM25 scores to [0, 1]
2. Weighted linear combination: final_score = w_vector * norm_vector + w_bm25 * norm_bm25
"""

from typing import Any, Dict, List, Optional, Tuple
import asyncio
import math

from pydantic import BaseModel, Field

from knowledge_base.persistence.v1.vector_store import VectorStore
from knowledge_base.storage.bm25_index import BM25Index, IndexedDocument, SearchResult
from knowledge_base.config.constants import (
    VECTOR_SEARCH_WEIGHT,
    GRAPH_SEARCH_WEIGHT,
    DEFAULT_TOP_K_RESULTS,
)


class HybridSearchResult(BaseModel):
    """Result from hybrid search combining vector and BM25.

    Attributes:
        id: Document/chunk ID.
        text: Text content.
        vector_score: Normalized vector similarity score.
        bm25_score: Normalized BM25 relevance score.
        final_score: Combined fusion score.
        metadata: Optional metadata.
        source: Source of the result (vector, bm25, or hybrid).
    """

    id: str = Field(..., description="Document or chunk ID")
    text: str = Field(..., description="Text content")
    vector_score: float = Field(default=0.0, description="Normalized vector similarity")
    bm25_score: float = Field(default=0.0, description="Normalized BM25 score")
    final_score: float = Field(..., description="Combined fusion score")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Document metadata"
    )
    source: str = Field(
        default="hybrid", description="Result source: vector, bm25, or hybrid"
    )


class HybridSearchEngine:
    """Hybrid search engine combining vector similarity + BM25.

    This class provides a unified search interface that executes both vector
    similarity search and BM25 keyword search in parallel, then fuses the
    results using weighted score combination.

    The fusion process:
    1. Execute vector and BM25 searches in parallel
    2. Normalize scores using min-max scaling
    3. Combine scores: final = w_vector * norm_vector + w_bm25 * norm_bm25
    4. Sort by final score and return top-k results

    Example:
        >>> engine = HybridSearchEngine(vector_store, bm25_index)
        >>> results = await engine.search(
        ...     query="machine learning algorithms",
        ...     vector_weight=0.7,
        ...     bm25_weight=0.3,
        ...     top_k=10,
        ... )
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
    ) -> None:
        """Initialize the hybrid search engine.

        Args:
            vector_store: VectorStore instance for similarity search.
            bm25_index: BM25Index instance for keyword search.

        Raises:
            ValueError: If vector_store or bm25_index is None.
        """
        if vector_store is None:
            raise ValueError("vector_store cannot be None")
        if bm25_index is None:
            raise ValueError("bm25_index cannot be None")

        self._vector_store = vector_store
        self._bm25_index = bm25_index

    async def search(
        self,
        query: str,
        vector_weight: float = VECTOR_SEARCH_WEIGHT,
        bm25_weight: float = GRAPH_SEARCH_WEIGHT,
        top_k: int = DEFAULT_TOP_K_RESULTS,
        filters: Optional[Dict] = None,
        domain: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[HybridSearchResult]:
        """Execute parallel vector + BM25 search with weighted fusion.

        Performs both vector similarity search and BM25 keyword search
        concurrently, then combines the results using weighted score fusion.

        Args:
            query: Search query string.
            vector_weight: Weight for vector similarity scores (0.0-1.0).
            bm25_weight: Weight for BM25 scores (0.0-1.0).
            top_k: Maximum number of results to return.
            filters: Optional metadata filters applied to BM25 results.
            domain: Optional domain filter for vector search.
            query_embedding: Pre-computed query embedding. If None, will
                use the query string directly for BM25 only.

        Returns:
            List of HybridSearchResult objects ranked by final score.

        Raises:
            ValueError: If weights don't sum to 1.0 or top_k is invalid.
        """
        if abs(vector_weight + bm25_weight - 1.0) > 1e-6:
            raise ValueError("vector_weight and bm25_weight must sum to 1.0")

        if top_k < 1:
            raise ValueError("top_k must be positive")

        vector_results: List[Dict[str, Any]] = []
        bm25_results: List[SearchResult] = []

        search_tasks = []

        if query_embedding is not None:
            vector_task = self._vector_search(
                query_embedding=query_embedding,
                top_k=top_k * 2,
                domain=domain,
            )
            search_tasks.append(("vector", vector_task))

        bm25_task = self._bm25_search(
            query=query,
            top_k=top_k * 2,
            filters=filters,
        )
        search_tasks.append(("bm25", bm25_task))

        results = await asyncio.gather(
            *[task for _, task in search_tasks],
            return_exceptions=True,
        )

        for (search_type, _), result in zip(search_tasks, results):
            if isinstance(result, Exception):
                continue
            if search_type == "vector":
                vector_results = result
            elif search_type == "bm25":
                bm25_results = result

        return self._fuse_results(
            vector_results=vector_results,
            bm25_results=bm25_results,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            top_k=top_k,
        )

    async def _vector_search(
        self,
        query_embedding: List[float],
        top_k: int,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute vector similarity search.

        Args:
            query_embedding: Query embedding vector.
            top_k: Maximum results.
            domain: Optional domain filter.

        Returns:
            List of vector search results.
        """
        try:
            results = await self._vector_store.search_similar_chunks(
                query_embedding=query_embedding,
                limit=top_k,
                similarity_threshold=0.0,
            )
            return results
        except Exception:
            return []

    async def _bm25_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Execute BM25 keyword search.

        Args:
            query: Search query string.
            top_k: Maximum results.
            filters: Optional metadata filters.

        Returns:
            List of BM25 search results.
        """
        try:
            results = await self._bm25_index.search(
                query=query,
                top_k=top_k,
                filters=filters,
            )
            return results
        except Exception:
            return []

    def _fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[SearchResult],
        vector_weight: float,
        bm25_weight: float,
        top_k: int,
    ) -> List[HybridSearchResult]:
        """Fuse vector and BM25 results using weighted score combination.

        Normalizes scores from both methods to [0, 1] range using min-max
        scaling, then combines them with the specified weights.

        Args:
            vector_results: Vector search results.
            bm25_results: BM25 search results.
            vector_weight: Weight for vector scores.
            bm25_weight: Weight for BM25 scores.
            top_k: Maximum results to return.

        Returns:
            Fused results sorted by final score.
        """
        vector_max = self._get_max_vector_score(vector_results)
        bm25_max = self._get_max_bm25_score(bm25_results)

        combined_scores: Dict[str, Tuple[float, float, float, str]] = {}

        for result in vector_results:
            doc_id = str(result.get("id", ""))
            similarity = float(result.get("similarity", 0.0))
            norm_score = self._normalize_vector_score(similarity, vector_max)
            combined_scores[doc_id] = (
                norm_score,
                0.0,
                norm_score * vector_weight,
                "vector",
            )

        for result in bm25_results:
            doc_id = result.id
            if doc_id in combined_scores:
                vec_score, _, _, _ = combined_scores[doc_id]
                combined_scores[doc_id] = (
                    vec_score,
                    self._normalize_bm25_score(result.score, bm25_max),
                    vec_score * vector_weight
                    + self._normalize_bm25_score(result.score, bm25_max) * bm25_weight,
                    "hybrid",
                )
            else:
                combined_scores[doc_id] = (
                    0.0,
                    self._normalize_bm25_score(result.score, bm25_max),
                    self._normalize_bm25_score(result.score, bm25_max) * bm25_weight,
                    "bm25",
                )

        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1][2],
            reverse=True,
        )

        vector_results_dict = {r.get("id", ""): r for r in vector_results}
        bm25_results_dict = {r.id: r for r in bm25_results}

        return [
            HybridSearchResult(
                id=doc_id,
                text=vector_results_dict.get(doc_id, {}).get(
                    "text",
                    bm25_results_dict.get(
                        doc_id, SearchResult(id="", text="", score=0)
                    ).text,
                ),
                vector_score=scores[0],
                bm25_score=scores[1],
                final_score=scores[2],
                metadata=vector_results_dict.get(doc_id, {}).get("metadata"),
                source=scores[3],
            )
            for doc_id, scores in sorted_results[:top_k]
        ]

    def _get_max_vector_score(self, results: List[Dict[str, Any]]) -> float:
        """Get maximum vector similarity score from results.

        Args:
            results: Vector search results.

        Returns:
            Maximum similarity score, or 1.0 if no results.
        """
        if not results:
            return 1.0
        return max(float(r.get("similarity", 0.0)) for r in results)

    def _get_max_bm25_score(self, results: List[SearchResult]) -> float:
        """Get maximum BM25 score from results.

        Args:
            results: BM25 search results.

        Returns:
            Maximum BM25 score, or 1.0 if no results.
        """
        if not results:
            return 1.0
        return max(r.score for r in results)

    def _normalize_vector_score(self, score: float, max_score: float) -> float:
        """Normalize vector similarity score to [0, 1].

        Uses min-max normalization. If max_score is 0 or very small,
        returns 0 to avoid division by zero.

        Args:
            score: Raw similarity score.
            max_score: Maximum score in result set.

        Returns:
            Normalized score in [0, 1].
        """
        if max_score <= 0:
            return 0.0
        return max(0.0, min(1.0, score / max_score))

    def _normalize_bm25_score(self, score: float, max_score: float) -> float:
        """Normalize BM25 score to [0, 1].

        Uses min-max normalization with log-based scaling for BM25
        scores which can be unbounded. Also applies smoothing.

        Args:
            score: Raw BM25 score.
            max_score: Maximum score in result set.

        Returns:
            Normalized score in [0, 1].
        """
        if max_score <= 0:
            return 0.0

        normalized = score / max_score
        normalized = max(0.0, min(1.0, normalized))

        return normalized

    async def search_with_reranking(
        self,
        query: str,
        initial_top_k: int = 50,
        final_top_k: int = DEFAULT_TOP_K_RESULTS,
        vector_weight: float = VECTOR_SEARCH_WEIGHT,
        bm25_weight: float = GRAPH_SEARCH_WEIGHT,
        filters: Optional[Dict] = None,
        domain: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[HybridSearchResult]:
        """Hybrid search with cross-encoder reranking (placeholder for Phase 2).

        This method performs an expanded hybrid search and returns a larger
        set of candidates, intended for use with a cross-encoder reranker.

        Currently this is a placeholder that returns the same results as
        search(). In Phase 2, this should be enhanced with cross-encoder
        reranking capabilities.

        Args:
            query: Search query string.
            initial_top_k: Number of initial candidates to retrieve.
            final_top_k: Number of final results after reranking.
            vector_weight: Weight for vector similarity.
            bm25_weight: Weight for BM25 scores.
            filters: Optional metadata filters.
            domain: Optional domain filter.
            query_embedding: Pre-computed query embedding.

        Returns:
            List of reranked HybridSearchResult objects.

        Raises:
            ValueError: If initial_top_k < final_top_k.
        """
        if initial_top_k < final_top_k:
            raise ValueError("initial_top_k must be >= final_top_k")

        expanded_results = await self.search(
            query=query,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            top_k=initial_top_k,
            filters=filters,
            domain=domain,
            query_embedding=query_embedding,
        )

        if len(expanded_results) <= final_top_k:
            return expanded_results

        return expanded_results[:final_top_k]

    async def index_document_for_hybrid_search(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """Index document chunks for hybrid search.

        Convenience method to index a document's chunks into both the
        vector store and BM25 index.

        Args:
            document_id: Source document ID.
            chunks: List of chunk dictionaries with 'text', 'embedding', 'metadata'.
        """
        for chunk in chunks:
            text = chunk.get("text", "")
            embedding = chunk.get("embedding")
            metadata = chunk.get("metadata", {})

            await self._bm25_index.index_documents(
                [
                    IndexedDocument(
                        id=str(chunk.get("id", "")),
                        text=text,
                        document_id=document_id,
                        metadata=metadata,
                    )
                ]
            )

            if embedding:
                try:
                    await self._vector_store.update_chunk_embedding(
                        chunk_id=str(chunk.get("id", "")),
                        embedding=embedding,
                    )
                except Exception:
                    pass

    async def delete_document_from_hybrid_search(
        self,
        document_id: str,
    ) -> None:
        """Delete document from both vector and BM25 indexes.

        Args:
            document_id: Document ID to delete.
        """
        await self._bm25_index.delete_documents([document_id])

    async def get_stats(self) -> Dict[str, Any]:
        """Get hybrid search engine statistics.

        Returns:
            Dictionary containing statistics from both indexes.
        """
        bm25_stats = await self._bm25_index.get_stats()

        return {
            "bm25_index": bm25_stats,
            "vector_weight": VECTOR_SEARCH_WEIGHT,
            "bm25_weight": GRAPH_SEARCH_WEIGHT,
        }
