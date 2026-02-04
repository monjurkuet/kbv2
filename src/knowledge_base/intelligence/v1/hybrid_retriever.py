"""Hybrid retrieval combining vector similarity and graph relationship expansion."""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass
class RetrievedEntity:
    """Entity retrieved from hybrid search."""

    id: UUID
    name: str
    entity_type: str
    description: str | None
    properties: dict[str, Any] | None
    confidence: float
    vector_score: float
    graph_score: float | None = None
    graph_hops: int | None = None
    normalized_vector_score: float | None = None
    normalized_graph_score: float | None = None
    final_score: float = 0.0
    source: str = "vector"


@dataclass
class HybridRetrievalResult:
    """Result of hybrid entity retrieval."""

    query: str
    entities: list[RetrievedEntity] = field(default_factory=list)
    vector_results_count: int = 0
    graph_results_count: int = 0
    total_hops_traversed: int = 0


class HybridEntityRetriever:
    """Hybrid entity retriever combining vector similarity and graph expansion."""

    DEFAULT_VECTOR_WEIGHT: float = 0.6
    DEFAULT_GRAPH_WEIGHT: float = 0.4
    DEFAULT_VECTOR_LIMIT: int = 20
    DEFAULT_GRAPH_LIMIT: int = 10
    DEFAULT_MIN_CONFIDENCE: float = 0.5

    def __init__(
        self,
        vector_store: Any,
        graph_store: Any,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        graph_weight: float = DEFAULT_GRAPH_WEIGHT,
    ) -> None:
        """
        Initialize hybrid entity retriever.

        Args:
            vector_store: Vector store for similarity search.
            graph_store: Graph store for relationship expansion.
            vector_weight: Weight for vector search scores (0-1).
            graph_weight: Weight for graph expansion scores (0-1).
        """
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._vector_weight = vector_weight
        self._graph_weight = graph_weight

        if abs(vector_weight + graph_weight - 1.0) > 1e-6:
            raise ValueError("vector_weight and graph_weight must sum to 1.0")

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        domain: str | None = None,
        entity_types: list[str] | None = None,
        vector_limit: int = DEFAULT_VECTOR_LIMIT,
        graph_limit: int = DEFAULT_GRAPH_LIMIT,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        graph_depth: int = 2,
    ) -> HybridRetrievalResult:
        """
        Perform hybrid retrieval combining vector and graph search.

        Args:
            query: Query text for context.
            query_embedding: Query vector for similarity search.
            domain: Optional domain filter.
            entity_types: Optional list of entity types to filter.
            vector_limit: Maximum vector search results.
            graph_limit: Maximum graph expansion results.
            min_confidence: Minimum confidence threshold.
            graph_depth: Depth for graph traversal.

        Returns:
            HybridRetrievalResult with combined and ranked entities.
        """
        result = HybridRetrievalResult(query=query)

        vector_entities = await self._vector_search(
            query_embedding=query_embedding,
            limit=vector_limit,
            min_confidence=min_confidence,
            domain=domain,
        )
        result.vector_results_count = len(vector_entities)

        entity_map: dict[UUID, RetrievedEntity] = {}
        for ve in vector_entities:
            entity = RetrievedEntity(
                id=ve["id"],
                name=ve["name"],
                entity_type=ve.get("entity_type", ""),
                description=ve.get("description"),
                properties=ve.get("properties"),
                confidence=ve.get("confidence", 0.0),
                vector_score=ve.get("similarity", 0.0),
                source="vector",
            )
            entity.final_score = entity.vector_score * self._vector_weight
            entity_map[entity.id] = entity

        if vector_entities:
            expanded_entities = await self._graph_expand(
                entity_ids=[e["id"] for e in vector_entities],
                depth=graph_depth,
                min_confidence=min_confidence,
                entity_types=entity_types,
                limit=graph_limit,
            )
            result.graph_results_count = len(expanded_entities)

            for ge in expanded_entities:
                if ge.id not in entity_map:
                    entity = RetrievedEntity(
                        id=ge.id,
                        name=ge.name,
                        entity_type=ge.entity_type,
                        description=ge.description,
                        properties=ge.properties,
                        confidence=ge.confidence,
                        vector_score=0.0,
                        graph_score=ge.confidence,
                        graph_hops=ge.hops,
                        source="graph",
                    )
                    entity.final_score = entity.graph_score * self._graph_weight
                    entity_map[entity.id] = entity

        result.entities = self._fuse_and_rank(entity_map)
        result.total_hops_traversed = sum(
            e.graph_hops or 0 for e in result.entities if e.graph_hops
        )

        return result

    async def _vector_search(
        self,
        query_embedding: list[float],
        limit: int,
        min_confidence: float,
        domain: str | None,
    ) -> list[dict[str, Any]]:
        """
        Perform vector-based similarity search.

        Args:
            query_embedding: Query vector.
            limit: Maximum results.
            min_confidence: Minimum confidence threshold.
            domain: Optional domain filter.

        Returns:
            List of similar entities with scores.
        """
        results = await self._vector_store.search_similar_entities(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=min_confidence,
        )

        if domain and results:
            results = [
                r for r in results if r.get("properties", {}).get("domain") == domain
            ]

        return results

    async def _graph_expand(
        self,
        entity_ids: list[UUID],
        depth: int,
        min_confidence: float,
        entity_types: list[str] | None,
        limit: int,
    ) -> list["GraphEntity"]:
        """
        Expand search via graph relationships.

        Args:
            entity_ids: Starting entity IDs.
            depth: Traversal depth.
            min_confidence: Minimum confidence threshold.
            entity_types: Optional entity type filter.
            limit: Maximum results.

        Returns:
            List of discovered entities from graph traversal.
        """
        from knowledge_base.persistence.v1.graph_store import GraphTraversalDirection

        all_neighbors: list[GraphEntity] = []
        visited: set[UUID] = set(entity_ids)

        for entity_id in entity_ids:
            if len(all_neighbors) >= limit:
                break

            try:
                center, neighbors = await self._graph_store.get_entity_neighborhood(
                    entity_id=entity_id,
                    depth=depth,
                    direction=GraphTraversalDirection.BIDIRECTIONAL,
                    min_confidence=min_confidence,
                    max_nodes=limit,
                    node_types=entity_types,
                )

                for neighbor, edge in neighbors:
                    if neighbor.id not in visited:
                        visited.add(neighbor.id)
                        hops = self._calculate_hops(depth, edge)
                        all_neighbors.append(
                            GraphEntity(
                                id=neighbor.id,
                                name=neighbor.name,
                                entity_type=neighbor.entity_type,
                                description=neighbor.description,
                                properties=neighbor.properties,
                                confidence=edge.confidence,
                                hops=hops,
                            )
                        )
            except Exception:
                continue

        return all_neighbors[:limit]

    def _calculate_hops(self, depth: int, edge: Any) -> int:
        """Calculate hop count for graph traversal."""
        return depth

    def _fuse_and_rank(
        self,
        entity_map: dict[UUID, RetrievedEntity],
    ) -> list[RetrievedEntity]:
        """
        Fuse and rank entities from both sources.

        Args:
            entity_map: Map of entity IDs to retrieved entities.

        Returns:
            Sorted list of entities by final score.
        """
        normalized_vector = iter(
            self._normalize_scores(
                [entity.vector_score for entity in entity_map.values()]
            )
        )
        normalized_graph = self._normalize_scores(
            [
                entity.graph_score
                for entity in entity_map.values()
                if entity.graph_score is not None
            ]
        )

        graph_iter = iter(normalized_graph)
        for entity in entity_map.values():
            entity.normalized_vector_score = next(normalized_vector, 0.0)
            if entity.graph_score is not None:
                entity.normalized_graph_score = next(graph_iter, 0.0)
            else:
                entity.normalized_graph_score = 0.0

            if entity.graph_score is not None and entity.graph_score > 0:
                graph_contribution = entity.normalized_graph_score * self._graph_weight
                entity.final_score = (
                    entity.normalized_vector_score * self._vector_weight
                ) + graph_contribution
            else:
                entity.final_score = (
                    entity.normalized_vector_score * self._vector_weight
                )

        sorted_entities = sorted(
            entity_map.values(),
            key=lambda e: e.final_score,
            reverse=True,
        )

        return sorted_entities

    @staticmethod
    def _normalize_scores(scores: list[float]) -> list[float]:
        """Normalize scores using min-max scaling."""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score - min_score <= 1e-8:
            return [1.0 for _ in scores]
        return [(score - min_score) / (max_score - min_score) for score in scores]


class GraphEntity:
    """Entity discovered from graph traversal."""

    def __init__(
        self,
        id: UUID,
        name: str,
        entity_type: str,
        description: str | None,
        properties: dict[str, Any] | None,
        confidence: float,
        hops: int,
    ) -> None:
        self.id = id
        self.name = name
        self.entity_type = entity_type
        self.description = description
        self.properties = properties
        self.confidence = confidence
        self.hops = hops
