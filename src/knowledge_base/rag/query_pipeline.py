"""RAG Query Pipeline for Knowledge Base System.

This module provides the main query pipeline that combines:
1. Query understanding
2. Context retrieval (hybrid search)
3. Context building
4. Answer generation
5. Source attribution

Supports multiple RAG strategies:
- Standard RAG
- LightRAG (dual-level retrieval)
- HippoRAG (graph-enhanced)
- CRAG (corrective retrieval)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

from knowledge_base.ingestion.vision_client import VisionModelClient
from knowledge_base.rag.context_builder import ContextBuilder, ContextWindow
from knowledge_base.rag.answer_generator import AnswerGenerator, RAGAnswer
from knowledge_base.storage.portable.hybrid_search import (
    HybridSearchEngine,
    HybridSearchResult,
    HybridSearchPipeline,
)
from knowledge_base.storage.portable.sqlite_store import SQLiteStore
from knowledge_base.storage.portable.chroma_store import ChromaStore
from knowledge_base.storage.portable.kuzu_store import KuzuGraphStore

logger = logging.getLogger(__name__)


class RAGMode(str, Enum):
    """RAG query modes."""

    STANDARD = "standard"  # Basic vector + keyword search
    HYBRID = "hybrid"  # BM25 + vector with RRF
    DUAL_LEVEL = "dual_level"  # LightRAG-style
    GRAPH_ENHANCED = "graph_enhanced"  # HippoRAG-style
    CORRECTIVE = "corrective"  # CRAG-style


@dataclass
class QueryResult:
    """Complete result from a RAG query."""

    question: str
    answer: RAGAnswer
    mode: RAGMode
    retrieved_chunks: list[HybridSearchResult]
    context_window: ContextWindow
    processing_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGQueryPipeline:
    """Main RAG query pipeline.

    This pipeline orchestrates the complete RAG workflow:
    1. Query understanding and optional rewriting
    2. Context retrieval using hybrid search
    3. Context building and formatting
    4. Answer generation with LLM
    5. Optional self-correction and refinement

    Example:
        >>> pipeline = RAGQueryPipeline(
        ...     search_engine=search_engine,
        ...     vision_client=vision_client,
        ... )
        >>> result = await pipeline.query("What is the BTC price target?")
        >>> print(result.answer.answer)
    """

    def __init__(
        self,
        search_engine: HybridSearchEngine,
        vision_client: VisionModelClient,
        kuzu_store: Optional[KuzuGraphStore] = None,
        default_mode: RAGMode = RAGMode.HYBRID,
        max_context_tokens: int = 4000,
    ) -> None:
        """Initialize RAG query pipeline.

        Args:
            search_engine: Hybrid search engine.
            vision_client: Vision model client for LLM calls.
            kuzu_store: Optional Kuzu graph store for graph-enhanced mode.
            default_mode: Default RAG mode.
            max_context_tokens: Maximum tokens in context window.
        """
        self._search = search_engine
        self._kuzu = kuzu_store
        self._default_mode = default_mode

        # Initialize components
        self._context_builder = ContextBuilder(
            max_tokens=max_context_tokens,
            include_metadata=True,
        )
        self._answer_generator = AnswerGenerator(vision_client)

        # CRAG evaluator (optional)
        self._evaluator_enabled = False

    async def query(
        self,
        question: str,
        mode: Optional[RAGMode] = None,
        filters: Optional[dict[str, Any]] = None,
        max_chunks: int = 20,
    ) -> QueryResult:
        """Execute a RAG query.

        Args:
            question: User question.
            mode: RAG mode (uses default if not specified).
            filters: Optional metadata filters.
            max_chunks: Maximum chunks to retrieve.

        Returns:
            QueryResult with answer and metadata.
        """
        import time
        start_time = time.time()

        mode = mode or self._default_mode

        # 1. Get query embedding (for semantic search)
        query_embedding = await self._get_embedding(question)

        # 2. Retrieve context based on mode
        if mode == RAGMode.STANDARD:
            chunks = await self._standard_retrieval(question, query_embedding, filters, max_chunks)
        elif mode == RAGMode.HYBRID:
            chunks = await self._hybrid_retrieval(question, query_embedding, filters, max_chunks)
        elif mode == RAGMode.DUAL_LEVEL:
            chunks = await self._dual_level_retrieval(question, query_embedding, filters, max_chunks)
        elif mode == RAGMode.GRAPH_ENHANCED:
            chunks = await self._graph_enhanced_retrieval(question, query_embedding, filters, max_chunks)
        elif mode == RAGMode.CORRECTIVE:
            chunks = await self._corrective_retrieval(question, query_embedding, filters, max_chunks)
        else:
            chunks = await self._hybrid_retrieval(question, query_embedding, filters, max_chunks)

        # 3. Build context window
        context = self._context_builder.build(chunks)

        # 4. Generate answer
        answer = await self._answer_generator.generate(question, context)

        # 5. Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        return QueryResult(
            question=question,
            answer=answer,
            mode=mode,
            retrieved_chunks=chunks,
            context_window=context,
            processing_time_ms=processing_time,
            metadata={
                "mode": mode.value,
                "chunks_retrieved": len(chunks),
                "context_tokens": context.total_tokens,
            },
        )

    async def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding for text (placeholder for embedding client).

        Args:
            text: Text to embed.

        Returns:
            Embedding vector or None.
        """
        # This would be implemented with an embedding client
        # For now, return None to use keyword search only
        return None

    async def _standard_retrieval(
        self,
        question: str,
        query_embedding: Optional[list[float]],
        filters: Optional[dict[str, Any]],
        max_chunks: int,
    ) -> list[HybridSearchResult]:
        """Standard retrieval - keyword or vector only.

        Args:
            question: User question.
            query_embedding: Query embedding.
            filters: Metadata filters.
            max_chunks: Maximum chunks.

        Returns:
            List of retrieved chunks.
        """
        if query_embedding:
            # Use vector search only
            return await self._search.search(
                query=question,
                query_embedding=query_embedding,
                limit=max_chunks,
                filters=filters,
            )
        else:
            # Use keyword search only
            return await self._search.search(
                query=question,
                limit=max_chunks,
                filters=filters,
            )

    async def _hybrid_retrieval(
        self,
        question: str,
        query_embedding: Optional[list[float]],
        filters: Optional[dict[str, Any]],
        max_chunks: int,
    ) -> list[HybridSearchResult]:
        """Hybrid retrieval - BM25 + vector with RRF.

        Args:
            question: User question.
            query_embedding: Query embedding.
            filters: Metadata filters.
            max_chunks: Maximum chunks.

        Returns:
            List of retrieved chunks.
        """
        return await self._search.search(
            query=question,
            query_embedding=query_embedding,
            limit=max_chunks,
            filters=filters,
            expand_graph=False,
        )

    async def _dual_level_retrieval(
        self,
        question: str,
        query_embedding: Optional[list[float]],
        filters: Optional[dict[str, Any]],
        max_chunks: int,
    ) -> list[HybridSearchResult]:
        """Dual-level retrieval (LightRAG-style).

        Retrieves both low-level entities and high-level concepts.

        Args:
            question: User question.
            query_embedding: Query embedding.
            filters: Metadata filters.
            max_chunks: Maximum chunks.

        Returns:
            List of retrieved chunks.
        """
        # Use the HybridSearchPipeline for dual-level
        pipeline = HybridSearchPipeline(
            search_engine=self._search,
            kuzu_store=self._kuzu,
        )

        return await pipeline.search(
            query=question,
            mode="dual",
            limit=max_chunks,
        )

    async def _graph_enhanced_retrieval(
        self,
        question: str,
        query_embedding: Optional[list[float]],
        filters: Optional[dict[str, Any]],
        max_chunks: int,
    ) -> list[HybridSearchResult]:
        """Graph-enhanced retrieval (HippoRAG-style).

        Uses graph traversal to expand context.

        Args:
            question: User question.
            query_embedding: Query embedding.
            filters: Metadata filters.
            max_chunks: Maximum chunks.

        Returns:
            List of retrieved chunks with graph context.
        """
        # First, do hybrid search
        initial_results = await self._search.search(
            query=question,
            query_embedding=query_embedding,
            limit=max_chunks // 2,
            filters=filters,
            expand_graph=True,
        )

        if not self._kuzu:
            return initial_results

        # Find entities in top results
        all_entities = []
        for result in initial_results:
            all_entities.extend(result.entities)

        # Get unique entity IDs
        entity_ids = list(set(e.get("id") for e in all_entities if e.get("id")))

        # Traverse graph from entities
        expanded_chunks = []
        for entity_id in entity_ids[:5]:  # Limit expansions
            try:
                traversal = await self._kuzu.traverse(
                    start_entity_id=entity_id,
                    max_depth=2,
                    limit=10,
                )

                # Get chunks mentioning connected entities
                for node in traversal.get("nodes", []):
                    node_chunks = await self._kuzu.query("""
                        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {id: $entity_id})
                        RETURN c.id, c.document_id, c.text
                        LIMIT 3
                    """, {"entity_id": node.get("id")})

                    for record in node_chunks.records:
                        expanded_chunks.append(HybridSearchResult(
                            chunk_id=record.get("c.id"),
                            document_id=record.get("c.document_id"),
                            text=record.get("c.text"),
                            score=0.5,  # Lower score for graph-expanded results
                            metadata={"source": "graph_traversal"},
                        ))
            except Exception as e:
                logger.debug(f"Graph traversal failed for {entity_id}: {e}")

        # Combine and deduplicate
        all_chunks = initial_results + expanded_chunks
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                unique_chunks.append(chunk)

        return unique_chunks[:max_chunks]

    async def _corrective_retrieval(
        self,
        question: str,
        query_embedding: Optional[list[float]],
        filters: Optional[dict[str, Any]],
        max_chunks: int,
    ) -> list[HybridSearchResult]:
        """Corrective retrieval (CRAG-style).

        Evaluates retrieved documents and re-queries if needed.

        Args:
            question: User question.
            query_embedding: Query embedding.
            filters: Metadata filters.
            max_chunks: Maximum chunks.

        Returns:
            List of evaluated and potentially refined chunks.
        """
        # Initial retrieval
        initial_results = await self._search.search(
            query=question,
            query_embedding=query_embedding,
            limit=max_chunks * 2,  # Get more for filtering
            filters=filters,
        )

        # Evaluate relevance
        relevant = []
        ambiguous = []

        for result in initial_results:
            # Simple relevance check based on score
            if result.score >= 0.7:
                relevant.append(result)
            elif result.score >= 0.4:
                ambiguous.append(result)

        # If we have enough high-quality results, return them
        if len(relevant) >= max_chunks // 2:
            return relevant[:max_chunks]

        # Otherwise, try query refinement
        if len(relevant) < max_chunks // 2:
            # Generate refined query
            refined_query = await self._refine_query(question, relevant, ambiguous)

            if refined_query and refined_query != question:
                # Re-query with refined question
                refined_results = await self._search.search(
                    query=refined_query,
                    query_embedding=query_embedding,
                    limit=max_chunks,
                    filters=filters,
                )

                # Combine results
                all_results = relevant + refined_results
                seen = set()
                unique = []
                for r in all_results:
                    if r.chunk_id not in seen:
                        seen.add(r.chunk_id)
                        unique.append(r)

                return unique[:max_chunks]

        # Return what we have
        return (relevant + ambiguous)[:max_chunks]

    async def _refine_query(
        self,
        question: str,
        relevant: list[HybridSearchResult],
        ambiguous: list[HybridSearchResult],
    ) -> Optional[str]:
        """Refine query based on retrieved results.

        Args:
            question: Original question.
            relevant: Relevant results.
            ambiguous: Ambiguous results.

        Returns:
            Refined query or None.
        """
        # This would use an LLM to refine the query
        # For now, return the original question
        return question

    async def query_with_reasoning(
        self,
        question: str,
        mode: Optional[RAGMode] = None,
    ) -> QueryResult:
        """Query with step-by-step reasoning.

        Args:
            question: User question.
            mode: RAG mode.

        Returns:
            QueryResult with reasoning included.
        """
        import time
        start_time = time.time()

        mode = mode or self._default_mode

        # Get chunks
        query_embedding = await self._get_embedding(question)
        chunks = await self._hybrid_retrieval(question, query_embedding, None, 20)

        # Build context
        context = self._context_builder.build(chunks)

        # Generate answer with reasoning
        answer = await self._answer_generator.generate_with_reasoning(question, context)

        processing_time = (time.time() - start_time) * 1000

        return QueryResult(
            question=question,
            answer=answer,
            mode=mode,
            retrieved_chunks=chunks,
            context_window=context,
            processing_time_ms=processing_time,
        )

    async def multi_query(
        self,
        questions: list[str],
        mode: Optional[RAGMode] = None,
    ) -> list[QueryResult]:
        """Process multiple queries.

        Args:
            questions: List of questions.
            mode: RAG mode.

        Returns:
            List of query results.
        """
        tasks = [self.query(q, mode) for q in questions]
        return await asyncio.gather(*tasks)
