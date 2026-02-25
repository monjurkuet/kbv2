"""Hybrid search engine combining BM25 and vector search.

This module implements a hybrid search strategy that combines:
- BM25 (via FTS5) for keyword-based search
- Vector similarity search for semantic search
- Graph-based context expansion via Kuzu
- Reciprocal Rank Fusion (RRF) for result combination

The hybrid approach provides better retrieval quality than either method alone.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from knowledge_base.storage.portable.config import HybridSearchConfig
from knowledge_base.storage.portable.sqlite_store import SearchResult, SQLiteStore
from knowledge_base.storage.portable.chroma_store import ChromaStore, EmbeddingResult

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0
    document_name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    entities: list[dict[str, Any]] = field(default_factory=list)


class HybridSearchEngine:
    """Hybrid search engine combining multiple search strategies.

    This engine combines:
    1. BM25 full-text search (via SQLite FTS5)
    2. Vector similarity search (via ChromaDB or sqlite-vec)
    3. Graph context expansion (via Kuzu)
    4. Reciprocal Rank Fusion for result combination

    Example:
        >>> engine = HybridSearchEngine(sqlite_store, chroma_store, kuzu_store)
        >>> results = await engine.search("bitcoin price target", limit=10)
    """

    def __init__(
        self,
        sqlite_store: SQLiteStore,
        chroma_store: Optional[ChromaStore] = None,
        kuzu_store: Optional[Any] = None,  # KuzuGraphStore
        config: Optional[HybridSearchConfig] = None,
    ) -> None:
        """Initialize hybrid search engine.

        Args:
            sqlite_store: SQLite store for FTS5 search.
            chroma_store: Optional ChromaDB store for vector search.
            kuzu_store: Optional Kuzu store for graph context.
            config: Hybrid search configuration.
        """
        self._sqlite = sqlite_store
        self._chroma = chroma_store
        self._kuzu = kuzu_store
        self._config = config or HybridSearchConfig()

    async def search(
        self,
        query: str,
        query_embedding: Optional[list[float]] = None,
        limit: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        expand_graph: bool = False,
    ) -> list[HybridSearchResult]:
        """Perform hybrid search.

        Args:
            query: Search query text.
            query_embedding: Optional pre-computed query embedding.
            limit: Maximum results. Uses config default if not provided.
            filters: Optional metadata filters.
            expand_graph: Whether to expand results with graph context.

        Returns:
            List of hybrid search results.
        """
        limit = limit or self._config.default_limit

        # Run searches in parallel
        tasks = []

        # BM25 search
        tasks.append(self._bm25_search(query, limit * 2))

        # Vector search (if available)
        vector_task = None
        if self._chroma and query_embedding:
            vector_task = self._vector_search(query_embedding, limit * 2, filters)
            tasks.append(vector_task)
        elif self._sqlite._vector_enabled and query_embedding:
            tasks.append(self._sqlite_vector_search(query_embedding, limit * 2))

        # Execute searches
        results = await asyncio.gather(*tasks, return_exceptions=True)

        bm25_results = results[0] if not isinstance(results[0], Exception) else []
        vector_results = (
            results[1] if len(results) > 1 and not isinstance(results[1], Exception) else []
        )

        # Combine with RRF
        combined = self._reciprocal_rank_fusion(bm25_results, vector_results, limit)

        # Expand with graph context if requested
        if expand_graph and self._kuzu:
            combined = await self._expand_with_graph(combined)

        return combined

    async def _bm25_search(self, query: str, limit: int) -> list[SearchResult]:
        """Perform BM25 search.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of search results.
        """
        try:
            return await self._sqlite.search_fts(query, limit)
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    async def _vector_search(
        self,
        query_embedding: list[float],
        limit: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[EmbeddingResult]:
        """Perform vector search via ChromaDB.

        Args:
            query_embedding: Query vector.
            limit: Maximum results.
            filters: Metadata filters.

        Returns:
            List of embedding results.
        """
        if not self._chroma:
            return []

        try:
            return await self._chroma.search(
                query_embedding,
                limit=limit,
                where=filters,
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _sqlite_vector_search(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> list[SearchResult]:
        """Perform vector search via sqlite-vec.

        Args:
            query_embedding: Query vector.
            limit: Maximum results.

        Returns:
            List of search results.
        """
        try:
            return await self._sqlite.search_vector(query_embedding, limit)
        except Exception as e:
            logger.error(f"SQLite vector search failed: {e}")
            return []

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[SearchResult],
        vector_results: list,
        limit: int,
    ) -> list[HybridSearchResult]:
        """Combine results using Reciprocal Rank Fusion.

        RRF formula: score(d) = sum(1 / (k + rank(d))) for each ranking

        Args:
            bm25_results: BM25 search results.
            vector_results: Vector search results.
            limit: Maximum results to return.

        Returns:
            Combined and ranked results.
        """
        k = self._config.rrf_k
        scores: dict[str, dict[str, Any]] = {}

        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            chunk_id = result.chunk_id
            if chunk_id not in scores:
                scores[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document_id": result.document_id,
                    "text": result.text,
                    "document_name": result.document_name,
                    "metadata": result.metadata,
                    "bm25_score": 0.0,
                    "vector_score": 0.0,
                    "rrf_score": 0.0,
                }

            rrf_contribution = 1 / (k + rank + 1)
            scores[chunk_id]["bm25_score"] = result.score
            scores[chunk_id]["rrf_score"] += rrf_contribution * self._config.bm25_weight

        # Process vector results
        for rank, result in enumerate(vector_results):
            chunk_id = result.chunk_id
            if chunk_id not in scores:
                scores[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document_id": getattr(result, "document_id", None),
                    "text": getattr(result, "text", None),
                    "document_name": None,
                    "metadata": getattr(result, "metadata", {}),
                    "bm25_score": 0.0,
                    "vector_score": 0.0,
                    "rrf_score": 0.0,
                }

            rrf_contribution = 1 / (k + rank + 1)
            scores[chunk_id]["vector_score"] = result.score
            scores[chunk_id]["rrf_score"] += rrf_contribution * self._config.vector_weight

        # Sort by RRF score and filter by minimum score
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        # Convert to HybridSearchResult
        results = []
        for item in sorted_results[:limit]:
            if item["rrf_score"] >= self._config.min_score:
                results.append(
                    HybridSearchResult(
                        chunk_id=item["chunk_id"],
                        document_id=item["document_id"],
                        text=item["text"],
                        score=item["rrf_score"],
                        bm25_score=item["bm25_score"],
                        vector_score=item["vector_score"],
                        document_name=item["document_name"],
                        metadata=item["metadata"],
                    )
                )

        return results

    async def _expand_with_graph(
        self,
        results: list[HybridSearchResult],
    ) -> list[HybridSearchResult]:
        """Expand results with graph context.

        For each result, fetch related entities from the knowledge graph.

        Args:
            results: Search results to expand.

        Returns:
            Results with entity information.
        """
        if not self._kuzu:
            return results

        expanded_results = []

        for result in results:
            try:
                # Get entities mentioned in this chunk
                entities = await self._kuzu.get_chunk_entities(result.chunk_id)

                result.entities = [
                    {
                        "id": e.id,
                        "name": e.name,
                        "type": e.entity_type,
                        "confidence": e.confidence,
                    }
                    for e in entities
                ]
            except Exception as e:
                logger.debug(f"Failed to get entities for chunk {result.chunk_id}: {e}")

            expanded_results.append(result)

        return expanded_results

    async def multi_query_search(
        self,
        queries: list[str],
        query_embeddings: Optional[list[list[float]]] = None,
        limit: int = 10,
    ) -> list[HybridSearchResult]:
        """Search with multiple queries (query expansion).

        This method searches with multiple query variations and
        combines the results.

        Args:
            queries: List of query variations.
            query_embeddings: Optional pre-computed embeddings.
            limit: Maximum results.

        Returns:
            Combined search results.
        """
        tasks = []
        for i, query in enumerate(queries):
            embedding = (
                query_embeddings[i] if query_embeddings and i < len(query_embeddings) else None
            )
            tasks.append(self.search(query, embedding, limit=limit * 2))

        # Execute all searches
        all_results = await asyncio.gather(*tasks)

        # Combine results with RRF
        combined_scores: dict[str, HybridSearchResult] = {}

        for results in all_results:
            for rank, result in enumerate(results):
                if result.chunk_id not in combined_scores:
                    combined_scores[result.chunk_id] = result

                # Add RRF contribution
                rrf = 1 / (self._config.rrf_k + rank + 1)
                combined_scores[result.chunk_id].score += rrf

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x.score,
            reverse=True,
        )

        return sorted_results[:limit]

    async def filtered_search(
        self,
        query: str,
        query_embedding: Optional[list[float]] = None,
        domain: Optional[str] = None,
        document_ids: Optional[list[str]] = None,
        date_range: Optional[tuple[str, str]] = None,
        limit: int = 10,
    ) -> list[HybridSearchResult]:
        """Search with metadata filters.

        Args:
            query: Search query.
            query_embedding: Query embedding.
            domain: Filter by domain.
            document_ids: Filter by document IDs.
            date_range: Filter by date range (start, end).
            limit: Maximum results.

        Returns:
            Filtered search results.
        """
        # Build filters for ChromaDB
        filters = {}
        if domain:
            filters["domain"] = domain
        if document_ids:
            filters["document_id"] = {"$in": document_ids}
        if date_range:
            filters["created_at"] = {
                "$gte": date_range[0],
                "$lte": date_range[1],
            }

        # Perform search with filters
        results = await self.search(
            query=query,
            query_embedding=query_embedding,
            limit=limit,
            filters=filters if filters else None,
        )

        # Additional filtering for SQLite results if needed
        if domain or document_ids:
            filtered = []
            for result in results:
                if domain and result.metadata.get("domain") != domain:
                    continue
                if document_ids and result.document_id not in document_ids:
                    continue
                filtered.append(result)
            results = filtered[:limit]

        return results


class HybridSearchPipeline:
    """Complete hybrid search pipeline with LightRAG-style dual-level retrieval.

    This pipeline implements:
    1. Query understanding and decomposition
    2. Low-level entity retrieval
    3. High-level concept retrieval
    4. Graph traversal for context expansion
    5. RAG answer generation

    This follows the LightRAG pattern for dual-level retrieval.
    """

    def __init__(
        self,
        search_engine: HybridSearchEngine,
        kuzu_store: Optional[Any] = None,
        embedding_client: Optional[Any] = None,
    ) -> None:
        """Initialize the search pipeline.

        Args:
            search_engine: Hybrid search engine.
            kuzu_store: Kuzu graph store for graph queries.
            embedding_client: Client for generating embeddings.
        """
        self._engine = search_engine
        self._kuzu = kuzu_store
        self._embedder = embedding_client

    async def search(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 10,
    ) -> list[HybridSearchResult]:
        """Search with specified mode.

        Args:
            query: Search query.
            mode: Search mode - 'keyword', 'semantic', 'hybrid', or 'dual'.
            limit: Maximum results.

        Returns:
            Search results.
        """
        # Get query embedding if semantic search needed
        query_embedding = None
        if mode in ["semantic", "hybrid", "dual"] and self._embedder:
            query_embedding = await self._get_embedding(query)

        if mode == "keyword":
            return await self._engine.search(query, limit=limit)
        elif mode == "semantic":
            return await self._engine.search(
                query, query_embedding, limit=limit, expand_graph=False
            )
        elif mode == "dual":
            return await self._dual_level_search(query, query_embedding, limit)
        else:  # hybrid
            return await self._engine.search(query, query_embedding, limit=limit, expand_graph=True)

    async def _dual_level_search(
        self,
        query: str,
        query_embedding: Optional[list[float]],
        limit: int,
    ) -> list[HybridSearchResult]:
        """Perform dual-level (LightRAG-style) search.

        This performs:
        1. Low-level: Entity retrieval
        2. High-level: Concept/community retrieval
        3. Graph traversal to connect them

        Args:
            query: Search query.
            query_embedding: Query embedding.
            limit: Maximum results.

        Returns:
            Dual-level search results.
        """
        results = []

        # Low-level: Search for specific entities and chunks
        low_level = await self._engine.search(
            query, query_embedding, limit=limit, expand_graph=True
        )
        results.extend(low_level)

        # High-level: Search for concepts/communities
        if self._kuzu:
            try:
                # Search for relevant communities
                community_results = await self._search_communities(query, limit // 2)

                # Get entities from top communities
                for community in community_results[:3]:
                    entities = await self._kuzu.get_community_entities(community["id"])
                    # Get chunks that mention these entities
                    for entity in entities[:5]:
                        entity_chunks = await self._get_entity_chunks(entity.id)
                        for chunk_data in entity_chunks:
                            # Add to results if not already present
                            if not any(r.chunk_id == chunk_data["id"] for r in results):
                                results.append(
                                    HybridSearchResult(
                                        chunk_id=chunk_data["id"],
                                        document_id=chunk_data["document_id"],
                                        text=chunk_data["text"],
                                        score=0.5,  # Lower score for high-level results
                                        metadata={
                                            "source": "community",
                                            "community": community["name"],
                                        },
                                    )
                                )
            except Exception as e:
                logger.error(f"High-level search failed: {e}")

        # Re-rank combined results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def _search_communities(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Search for relevant communities by name/summary.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of matching communities.
        """
        if not self._kuzu:
            return []

        try:
            result = await self._kuzu.query(
                """
                MATCH (c:Community)
                WHERE c.name CONTAINS $query OR c.summary CONTAINS $query
                RETURN c.id, c.name, c.summary, c.entity_count
                ORDER BY c.entity_count DESC
                LIMIT $limit
            """,
                {"query": query, "limit": limit},
            )

            return [
                {
                    "id": r.get("c.id"),
                    "name": r.get("c.name"),
                    "summary": r.get("c.summary"),
                    "entity_count": r.get("c.entity_count"),
                }
                for r in result.records
            ]
        except Exception as e:
            logger.error(f"Community search failed: {e}")
            return []

    async def _get_entity_chunks(self, entity_id: str) -> list[dict[str, Any]]:
        """Get chunks that mention an entity.

        Args:
            entity_id: Entity ID.

        Returns:
            List of chunks mentioning the entity.
        """
        if not self._kuzu:
            return []

        try:
            result = await self._kuzu.query(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {id: $entity_id})
                RETURN c.id, c.document_id, c.text
                LIMIT 5
            """,
                {"entity_id": entity_id},
            )

            return [
                {
                    "id": r.get("c.id"),
                    "document_id": r.get("c.document_id"),
                    "text": r.get("c.text"),
                }
                for r in result.records
            ]
        except Exception as e:
            logger.error(f"Failed to get entity chunks: {e}")
            return []

    async def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        if not self._embedder:
            return None

        try:
            if hasattr(self._embedder, "embed_query"):
                return await self._embedder.embed_query(text)
            elif hasattr(self._embedder, "embed"):
                embedding = self._embedder.embed(text)
                return embedding.tolist() if hasattr(embedding, "tolist") else embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")

        return None
