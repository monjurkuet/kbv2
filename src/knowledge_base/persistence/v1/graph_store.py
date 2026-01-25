"""
Graph repository for graph visualization and analysis operations.

This module provides GraphStore, a repository pattern implementation for
complex graph operations including community aggregation, neighborhood traversal,
and inter-community edge calculation following Google AIP standards.
"""

from typing import List, Tuple, Dict, Any, Optional
from uuid import UUID
from enum import Enum

from sqlalchemy import select, func, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from knowledge_base.persistence.v1.schema import (
    Entity,
    Edge,
    Community,
    Document,
    Chunk,
)
from knowledge_base.common.api_models import PaginatedResponse


class GraphTraversalDirection(str, Enum):
    """Direction of graph traversal for neighborhood queries."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BIDIRECTIONAL = "bidirectional"


class EdgeAggregationResult:
    """
    Result of aggregating edges between communities.

    Represents a weighted edge connecting two communities in the graph summary view.
    """

    def __init__(
        self,
        source: UUID,
        target: UUID,
        weight: int,
        avg_confidence: float,
        edge_types: List[str],
    ):
        """
        Initialize edge aggregation result.

        Args:
            source: Source community UUID
            target: Target community UUID
            weight: Number of edges between these communities
            avg_confidence: Average confidence score of aggregated edges
            edge_types: List of edge types in the aggregation
        """
        self.source = source
        self.target = target
        self.weight = weight
        self.avg_confidence = avg_confidence
        self.edge_types = edge_types


class EntityWithEdges:
    """
    Entity with its connected edges for neighborhood expansion.

    Used to efficiently fetch both entities and their relationships
    in a single database round-trip.
    """

    def __init__(self, entity: Entity, edges: List[Edge]):
        """
        Initialize entity with edges.

        Args:
            entity: The entity instance
            edges: List of edges connected to this entity
        """
        self.entity = entity
        self.edges = edges


class GraphStore:
    """
    Repository for graph-specific database operations.

    Encapsulates complex SQL queries for graph visualization and analysis,
    following the repository pattern to separate database logic from API layer.

    Key Responsibilities:
    - Community topology aggregation (Map-Reduce style)
    - Neighborhood traversal and expansion
    - Path finding algorithms
    - Temporal graph analysis
    - Graph statistics and metrics
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize GraphStore with database session.

        Args:
            db: Async SQLAlchemy session for database operations
        """
        self.db = db

    async def get_community_topology(
        self, level: int = 0, min_community_size: int = 3, domain: str | None = None
    ) -> Tuple[List[Community], List[EdgeAggregationResult]]:
        """
        Get community-level graph topology with inter-community edges.

        This implements the "Map-Reduce" style aggregation described in the
        high-level logic guide. It efficiently calculates edges between
        communities without fetching all individual entity edges.

        Args:
            level: Community hierarchy level (0=macro, 1=micro)
            min_community_size: Minimum entities required in a community
            domain: Optional domain filter for multi-tenancy

        Returns:
            Tuple of (communities, aggregated_edges)
        """
        # Fetch communities at specified level
        community_query = select(Community).where(Community.level == level)

        if min_community_size > 0:
            community_query = community_query.where(
                Community.entity_count >= min_community_size
            )

        if domain:
            community_query = community_query.where(Community.domain == domain)

        result = await self.db.execute(community_query)
        communities = result.scalars().all()

        if not communities:
            return [], []

        # Aggregate inter-community edges using efficient SQL
        # This is the critical "Map-Reduce" operation from the logic guide
        community_ids = [c.id for c in communities]

        edge_agg_query = text("""
            SELECT 
                s.community_id as source_community_id,
                t.community_id as target_community_id,
                COUNT(*) as edge_count,
                AVG(e.confidence) as avg_confidence,
                ARRAY_AGG(DISTINCT e.edge_type) as edge_types
            FROM edges e
            JOIN entities s ON e.source_id = s.id
            JOIN entities t ON e.target_id = t.id
            WHERE s.community_id != t.community_id
                AND s.community_id = ANY(:community_ids)
                AND t.community_id = ANY(:community_ids)
            GROUP BY s.community_id, t.community_id
            HAVING COUNT(*) > 0
        """)

        result = await self.db.execute(edge_agg_query, {"community_ids": community_ids})

        aggregated_edges = []
        for row in result.fetchall():
            aggregated_edges.append(
                EdgeAggregationResult(
                    source=row.source_community_id,
                    target=row.target_community_id,
                    weight=row.edge_count,
                    avg_confidence=row.avg_confidence or 0.0,
                    edge_types=row.edge_types or [],
                )
            )

        return communities, aggregated_edges

    async def get_entity_neighborhood(
        self,
        entity_id: UUID,
        depth: int = 1,
        direction: GraphTraversalDirection = GraphTraversalDirection.BIDIRECTIONAL,
        min_confidence: float = 0.7,
        max_nodes: int = 1000,
        node_types: List[str] | None = None,
        edge_types: List[str] | None = None,
    ) -> Tuple[Entity, List[Tuple[Entity, Edge]]]:
        """
        Get entity neighborhood with connected edges and neighbors.

        Implements the neighborhood drill-down logic from the high-level guide.
        Fetches the center entity, its immediate edges, and all connected
        neighbor entities in an efficient batched query.

        Args:
            entity_id: Center entity UUID
            depth: Traversal depth (currently supports depth=1, depth>1 requires recursive CTE)
            direction: Edge direction to follow (outgoing, incoming, bidirectional)
            min_confidence: Minimum confidence threshold for edges
            max_nodes: Maximum nodes to return (safety limit)
            node_types: Optional filter for entity types
            edge_types: Optional filter for relationship types

        Returns:
            Tuple of (center_entity, list_of(neighbor, edge))
        """
        # Fetch center entity
        entity_query = select(Entity).where(Entity.id == entity_id)
        result = await self.db.execute(entity_query)
        center_entity = result.scalar_one_or_none()

        if not center_entity:
            raise ValueError(f"Entity {entity_id} not found")

        # Build edge query based on direction
        if direction == GraphTraversalDirection.OUTGOING:
            edge_filter = Edge.source_id == entity_id
        elif direction == GraphTraversalDirection.INCOMING:
            edge_filter = Edge.target_id == entity_id
        else:  # BIDIRECTIONAL
            edge_filter = or_(Edge.source_id == entity_id, Edge.target_id == entity_id)

        # Add confidence threshold
        edge_query = select(Edge).where(
            and_(edge_filter, Edge.confidence >= min_confidence)
        )

        # Add edge type filter
        if edge_types:
            edge_query = edge_query.where(Edge.edge_type.in_(edge_types))

        # Apply max nodes limit
        edge_query = edge_query.limit(max_nodes * 2)  # *2 for bidirectional

        result = await self.db.execute(edge_query)
        edges = result.scalars().all()

        if not edges:
            return center_entity, []

        # Collect all neighbor IDs
        neighbor_ids = set()
        for edge in edges:
            if edge.source_id == entity_id:
                neighbor_ids.add(edge.target_id)
            if edge.target_id == entity_id:
                neighbor_ids.add(edge.source_id)

        # Remove center entity from neighbors
        neighbor_ids.discard(entity_id)

        # Fetch all neighbor entities in one batch
        if neighbor_ids:
            entity_query = select(Entity).where(Entity.id.in_((neighbor_ids)))

            if node_types:
                entity_query = entity_query.where(Entity.entity_type.in_(node_types))

            result = await self.db.execute(entity_query)
            neighbor_entities = {e.id: e for e in result.scalars().all()}
        else:
            neighbor_entities = {}

        # Build result: list of (neighbor_entity, edge) tuples
        neighborhood = []
        for edge in edges:
            neighbor_id = None

            # Determine which side of the edge is the neighbor
            if edge.source_id == entity_id and edge.target_id in neighbor_entities:
                neighbor_id = edge.target_id
            elif edge.target_id == entity_id and edge.source_id in neighbor_entities:
                neighbor_id = edge.source_id

            if neighbor_id and neighbor_id in neighbor_entities:
                neighbor = neighbor_entities[neighbor_id]

                # Apply node filter if specified
                if node_types is None or neighbor.entity_type in node_types:
                    neighborhood.append((neighbor, edge))

        return center_entity, neighborhood

    async def get_temporal_trajectory(
        self,
        entity_ids: List[UUID],
        start_date: Any,  # datetime or similar
        end_date: Any,
        time_step: str = "month",
    ) -> List[Dict[str, Any]]:
        """
        Get temporal evolution of entity relationships over time.

        Implements the 2026 trajectory endpoint logic for time-based
        graph visualization with bitemporal support (valid time and transaction time).

        Args:
            entity_ids: Entities to track over time
            start_date: Start timestamp for analysis
            end_date: End timestamp for analysis
            time_step: Time resolution (day, week, month, year)

        Returns:
            List of temporal snapshots with graph state at each time point
        """
        # Query edges with temporal validity within the timeframe
        edge_query = (
            select(Edge)
            .where(
                and_(
                    or_(
                        Edge.source_id.in_((entity_ids)),
                        Edge.target_id.in_((entity_ids)),
                    ),
                    Edge.temporal_validity_start <= end_date,
                    or_(
                        Edge.temporal_validity_end.is_(None),
                        Edge.temporal_validity_end >= start_date,
                    ),
                )
            )
            .order_by(Edge.temporal_validity_start)
        )

        result = await self.db.execute(edge_query)
        edges = result.scalars().all()

        # Bucket edges by time step
        from datetime import datetime, timedelta

        trajectory = []
        current_time = start_date

        while current_time <= end_date:
            # Determine next time boundary
            if time_step == "day":
                next_time = current_time + timedelta(days=1)
            elif time_step == "week":
                next_time = current_time + timedelta(weeks=1)
            elif time_step == "month":
                # Simple month calculation
                if current_time.month == 12:
                    next_time = current_time.replace(
                        year=current_time.year + 1, month=1
                    )
                else:
                    next_time = current_time.replace(month=current_time.month + 1)
            else:  # year
                next_time = current_time.replace(year=current_time.year + 1)

            # Find edges valid during this time period
            valid_edges = []
            for edge in edges:
                edge_start = edge.temporal_validity_start
                edge_end = edge.temporal_validity_end

                # Check if edge was valid during this time bucket
                if edge_start <= current_time and (
                    edge_end is None or edge_end >= current_time
                ):
                    valid_edges.append(edge)

            # Build snapshot for this time period
            if valid_edges:
                snapshot = {
                    "timestamp": current_time.isoformat(),
                    "edges": valid_edges,
                    "edge_count": len(valid_edges),
                    "active_entities": set(),
                }

                # Collect all active entities
                for edge in valid_edges:
                    snapshot["active_entities"].add(edge.source_id)
                    snapshot["active_entities"].add(edge.target_id)

                snapshot["active_entities"] = list(snapshot["active_entities"])
                trajectory.append(snapshot)

            current_time = next_time

        return trajectory

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Calculate graph-level statistics and metrics.

        Returns:
            Dictionary with graph statistics
        """
        # Basic entity statistics
        entity_stats_query = select(
            func.count(Entity.id),
            func.count(Entity.id.distinct()).filter(Entity.entity_type.isnot(None)),
            func.avg(Entity.confidence),
        )

        result = await self.db.execute(entity_stats_query)
        entity_count, entity_type_count, avg_entity_confidence = result.one()

        # Edge statistics
        edge_stats_query = select(
            func.count(Edge.id),
            func.count(Edge.id.distinct()).filter(Edge.edge_type.isnot(None)),
            func.avg(Edge.confidence),
        )

        result = await self.db.execute(edge_stats_query)
        edge_count, edge_type_count, avg_edge_confidence = result.one()

        # Community statistics
        community_query = select(func.count(Community.id))
        result = await self.db.execute(community_query)
        community_count = result.scalar()

        return {
            "entity_count": entity_count or 0,
            "entity_types": entity_type_count or 0,
            "avg_entity_confidence": float(avg_entity_confidence or 0.0),
            "edge_count": edge_count or 0,
            "edge_types": edge_type_count or 0,
            "avg_edge_confidence": float(avg_edge_confidence or 0.0),
            "community_count": community_count or 0,
            "last_updated": None,  # Could add timestamp tracking
        }


# Helper function for dependency injection
def get_graph_store(db: AsyncSession) -> GraphStore:
    """
    Dependency factory for GraphStore instances.

    Args:
        db: Database session from dependency injection

    Returns:
        GraphStore instance ready for use in endpoints
    """
    return GraphStore(db)
