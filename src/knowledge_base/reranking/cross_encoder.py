"""Cross-encoder based reranking for improved search results.

This module provides a CrossEncoderReranker class that uses cross-encoder models
to score query-document pairs directly, enabling more accurate reranking of
search results compared to sparse (BM25) or dense (vector) methods alone.

Cross-encoder models process both the query and document together through a
transformer network, allowing for deep semantic understanding and more nuanced
scoring than bi-encoder approaches.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

from sentence_transformers import CrossEncoder
from pydantic import BaseModel, Field

from knowledge_base.storage.portable.hybrid_search import HybridSearchResult


logger = logging.getLogger(__name__)


class RerankedSearchResult(BaseModel):
    """Search result enhanced with cross-encoder reranking scores.

    Attributes:
        id: Document/chunk ID.
        text: Text content.
        vector_score: Normalized vector similarity score from hybrid search.
        bm25_score: Normalized BM25 score from hybrid search.
        final_score: Combined fusion score from hybrid search.
        cross_encoder_score: Cross-encoder relevance score.
        reranked_score: Final score after cross-encoder reranking.
        metadata: Optional metadata.
        source: Source of the result from hybrid search.
    """

    id: str = Field(..., description="Document or chunk ID")
    text: str = Field(..., description="Text content")
    vector_score: float = Field(default=0.0, description="Normalized vector similarity")
    bm25_score: float = Field(default=0.0, description="Normalized BM25 score")
    final_score: float = Field(default=0.0, description="Combined fusion score")
    cross_encoder_score: float = Field(default=0.0, description="Cross-encoder relevance score")
    reranked_score: float = Field(
        default=0.0, description="Final score after cross-encoder reranking"
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")
    source: str = Field(default="hybrid", description="Result source: vector, bm25, or hybrid")


class CrossEncoderReranker:
    """Cross-encoder based reranking for improved search results.

    This class uses pre-trained cross-encoder models to score query-document
    pairs directly. Cross-encoders provide more accurate relevance scoring
    than bi-encoders by jointly encoding query and document, though at the
    cost of slower inference speed.

    The cross-encoder scores are combined with hybrid search scores to produce
    final reranked results. The cross-encoder score typically dominates the
    final ranking since it captures deeper semantic relevance.

    Supported models (in order of quality/speed):
        - cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, good quality
        - cross-encoder/ms-marco-TinyBERT-L-4: Very fast, decent quality
        - cross-encoder/ms-marco-MiniLM-L-12-v2: Slower, better quality
        - cross-encoder/ms-marco-MiniLM-L-12-v2: Slower, better quality

    Example:
        >>> reranker = CrossEncoderReranker()
        >>> await reranker.initialize()
        >>> candidates = [HybridSearchResult(id="1", text="Python guide", ...)]
        >>> results = await reranker.rerank("Python tutorial", candidates)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ) -> None:
        """Initialize the cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model from HuggingFace Hub.
                Default is ms-marco-MiniLM-L-6-v2 which provides good
                quality/speed tradeoff for MS MARCO passage retrieval.
            device: Device to run the model on. If None, automatically
                selects CUDA if available, else CPU.

        Raises:
            ValueError: If model_name is empty or None.
        """
        if not model_name:
            raise ValueError("model_name cannot be empty")

        self._model_name = model_name
        self._device = device
        self._model: Optional[CrossEncoder] = None
        self._initialized = False

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def is_initialized(self) -> bool:
        """Check if the model is loaded."""
        return self._initialized

    async def initialize(self) -> None:
        """Load the cross-encoder model.

        This method loads the specified cross-encoder model from HuggingFace.
        The model is loaded once and cached for subsequent use.

        Note:
            This method is async to allow non-blocking model loading in
            server contexts. The actual model loading happens synchronously
            since sentence-transformers doesn't provide async loading.

        Raises:
            RuntimeError: If model loading fails.
        """
        if self._initialized:
            return

        logger.info(f"Loading cross-encoder model: {self._model_name}")

        try:
            import threading

            # Run model loading in a thread to avoid blocking
            model_loaded = threading.Event()

            def load_model() -> None:
                try:
                    self._model = CrossEncoder(
                        self._model_name,
                        device=self._device,
                        trust_remote_code=True,
                    )
                    model_loaded.set()
                except Exception as e:
                    logger.error(f"Failed to load cross-encoder model: {e}")
                    model_loaded.set()
                    raise

            load_model()
            self._initialized = True
            logger.info(f"Cross-encoder model loaded successfully: {self._model_name}")

        except Exception as e:
            raise RuntimeError(f"Failed to load cross-encoder model: {e}")

    async def rerank(
        self,
        query: str,
        candidates: List[HybridSearchResult],
        top_k: int = 5,
        batch_size: int = 32,
    ) -> List[RerankedSearchResult]:
        """Rerank candidates using cross-encoder scores.

        Scores each candidate document against the query using the cross-encoder,
        then sorts and returns the top-k results. The final score combines
        the cross-encoder score with the original hybrid search score.

        Args:
            query: Search query string.
            candidates: List of candidate search results to rerank.
            top_k: Number of top results to return after reranking.
            batch_size: Batch size for scoring documents. Larger batches
                are more efficient but use more memory.

        Returns:
            List of RerankedSearchResult objects sorted by reranked score.

        Raises:
            RuntimeError: If the model is not initialized.
            ValueError: If top_k is negative or batch_size is invalid.
        """
        if not self._initialized:
            raise RuntimeError("CrossEncoderReranker not initialized. Call initialize() first.")

        if top_k < 0:
            raise ValueError("top_k must be non-negative")

        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        if not candidates:
            return []

        documents = [cand.text for cand in candidates]

        ce_scores = await self.score_batch(query=query, documents=documents, batch_size=batch_size)

        reranked_results: List[RerankedSearchResult] = []
        for candidate, ce_score in zip(candidates, ce_scores):
            reranked_score = self._compute_reranked_score(
                hybrid_score=candidate.final_score, ce_score=ce_score
            )

            reranked_results.append(
                RerankedSearchResult(
                    id=candidate.id,
                    text=candidate.text,
                    vector_score=candidate.vector_score,
                    bm25_score=candidate.bm25_score,
                    final_score=candidate.final_score,
                    cross_encoder_score=ce_score,
                    reranked_score=reranked_score,
                    metadata=candidate.metadata,
                    source=candidate.source,
                )
            )

        reranked_results.sort(key=lambda x: x.reranked_score, reverse=True)

        return reranked_results[:top_k]

    async def rerank_with_scores(
        self,
        query: str,
        candidates: List[HybridSearchResult],
        top_k: int = 5,
    ) -> Tuple[List[RerankedSearchResult], List[float]]:
        """Rerank and return scores alongside results.

        Scores each candidate document and returns both the reranked results
        and the raw cross-encoder scores for analysis.

        Args:
            query: Search query string.
            candidates: List of candidate search results to rerank.
            top_k: Number of top results to return.

        Returns:
            Tuple of (reranked results, cross-encoder scores).

        Raises:
            RuntimeError: If the model is not initialized.
        """
        if not self._initialized:
            raise RuntimeError("CrossEncoderReranker not initialized. Call initialize() first.")

        if not candidates:
            return [], []

        documents = [cand.text for cand in candidates]
        ce_scores = await self.score_batch(query=query, documents=documents)

        results = await self.rerank(query=query, candidates=candidates, top_k=top_k)

        return results, ce_scores

    def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair.

        Synchronous method for scoring a single document against a query.
        Use score_batch for multiple documents to benefit from batching.

        Args:
            query: Search query string.
            document: Document text to score.

        Returns:
            Cross-encoder score (higher = more relevant).

        Raises:
            RuntimeError: If the model is not initialized.
        """
        if not self._initialized:
            raise RuntimeError("CrossEncoderReranker not initialized. Call initialize() first.")

        if not query or not document:
            return 0.0

        scores = self._model.predict([(query, document)])
        return float(scores[0])

    async def score_batch(
        self,
        query: str,
        documents: List[str],
        batch_size: int = 32,
    ) -> List[float]:
        """Score multiple documents for a query (async batch).

        Scores all documents against the query in batches for efficient
        processing. This is the recommended method for scoring multiple
        documents as it provides significant speedup through batching.

        Args:
            query: Search query string.
            documents: List of document texts to score.
            batch_size: Number of documents to score per batch.

        Returns:
            List of cross-encoder scores corresponding to input documents.

        Raises:
            RuntimeError: If the model is not initialized.
            ValueError: If documents is empty or batch_size is invalid.
        """
        if not self._initialized:
            raise RuntimeError("CrossEncoderReranker not initialized. Call initialize() first.")

        if not documents:
            return []

        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        pairs = [(query, doc) for doc in documents]

        all_scores: List[float] = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            scores = self._model.predict(batch)
            all_scores.extend([float(s) for s in scores])

        return all_scores

    def _compute_reranked_score(
        self, hybrid_score: float, ce_score: float, ce_weight: float = 0.7
    ) -> float:
        """Compute combined reranked score.

        Combines the hybrid search score with the cross-encoder score
        using weighted combination. The cross-encoder score typically
        gets higher weight since it captures deeper semantic relevance.

        Args:
            hybrid_score: Original hybrid search final_score.
            ce_score: Cross-encoder relevance score.
            ce_weight: Weight for cross-encoder score (0-1).

        Returns:
            Combined reranked score.
        """
        normalized_hybrid = min(1.0, hybrid_score)

        return ce_weight * ce_score + (1 - ce_weight) * normalized_hybrid

    async def shutdown(self) -> None:
        """Clean up model resources.

        Unloads the model and releases GPU memory if applicable.
        """
        if self._model is not None:
            del self._model
            self._model = None

        import gc

        gc.collect()

        self._initialized = False
        logger.info("Cross-encoder model unloaded")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"CrossEncoderReranker(model_name='{self._model_name}')"

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if hasattr(self, "_model") and self._model is not None:
            import gc

            del self._model
            gc.collect()
