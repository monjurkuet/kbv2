"""
Graph Visualization API following Google AIP standards and Pattern B endpoint structure.

This module provides endpoints for graph operations including community-level summaries,
neighborhood traversal, temporal analysis, and export functionality for Sigma.js integration.

Endpoints follow AIP-136 custom method pattern:
- GET /api/v1/graphs/{graph_id}:summary
- GET /api/v1/graphs/{graph_id}/nodes/{node_id}:neighborhood
- POST /api/v1/graphs/{graph_id}:findPath
- GET /api/v1/graphs/{graph_id}:export
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from enum import Enum
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_base.common.api_models import APIResponse
from knowledge_base.common.pagination import PageParams
from knowledge_base.persistence.v1.graph_store import (
    GraphStore,
    GraphTraversalDirection,
)
from knowledge_base.persistence.v1.schema import Entity, Edge


router = APIRouter(prefix="/api/v1/graphs", tags=["graph"])


class GraphNode(BaseModel):
    """Sigma.js compatible node representation."""

    key: str = Field(..., description="Unique node identifier (UUID)")
    label: str = Field(..., description="Display label for the node")
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Node attributes: x, y, size, color, community, entity_type, etc.",
    )


class GraphEdge(BaseModel):
    """Sigma.js compatible edge representation."""

    key: str = Field(..., description="Unique edge identifier (UUID)")
    source: str = Field(..., description="Source node key")
    target: str = Field(..., description="Target node key")
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Edge attributes: label, weight, type, confidence, color",
    )


class GraphSummaryResponse(BaseModel):
    """Response for community-level graph summary view."""

    nodes: List[GraphNode] = Field(..., description="Community nodes (level 0)")
    edges: List[GraphEdge] = Field(..., description="Aggregated inter-community edges")
    total_nodes: int = Field(..., description="Total number of nodes")
    total_edges: int = Field(..., description="Total number of edges")
    stats: Dict[str, Any] = Field(default_factory=dict, description="Graph statistics")


class NeighborhoodResponse(BaseModel):
    """Response for entity neighborhood expansion."""

    center_node: GraphNode = Field(..., description="Center entity node")
    nodes: List[GraphNode] = Field(..., description="Neighbor nodes including center")
    edges: List[GraphEdge] = Field(..., description="Edges connecting neighbors")
    expanded_count: int = Field(..., description="Number of new nodes added")
    max_confidence: float = Field(..., description="Maximum edge confidence in result")
    min_confidence: float = Field(..., description="Minimum edge confidence in result")


class PathFindingRequest(BaseModel):
    """Request model for graph path finding operations."""

    source: str = Field(..., description="Source node ID (graphs/{graph}/nodes/{node})")
    target: str = Field(..., description="Target node ID")
    algorithm: str = Field(
        default="shortest_path",
        pattern="^(shortest_path|all_simple_paths|most_confident)$",
        description="Path finding algorithm to use",
    )
    max_hops: int = Field(
        default=5, ge=1, le=10, description="Maximum number of hops allowed in path"
    )
    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for edges",
    )
    edge_types: Optional[List[str]] = Field(
        default=None, description="Optional filter for specific edge types"
    )


class PathResponse(BaseModel):
    """Response for path finding operations."""

    source: str = Field(..., description="Source node")
    target: str = Field(..., description="Target node")
    paths: List[List[str]] = Field(
        ..., description="List of paths (each path is list of node IDs)"
    )
    confidence_scores: List[float] = Field(
        ..., description="Confidence score for each path"
    )
    total_paths: int = Field(..., description="Total number of paths found")
    algorithm: str = Field(..., description="Algorithm used")


class GraphologyExport(BaseModel):
    """Graphology JSON format for Sigma.js compatibility."""

    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Graph-level attributes (name, description, etc.)",
    )
    options: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "directed", "multi": True},
        description="Graph options for Sigma.js",
    )
    nodes: List[GraphNode] = Field(..., description="All nodes in graph")
    edges: List[GraphEdge] = Field(..., description="All edges in graph")


async def get_db():
    """Dependency to get database session."""
    from knowledge_base.common.dependencies import get_async_session

    async for session in get_async_session():
        yield session


# Dependency injection for database session
def get_graph_store(db: AsyncSession = Depends(get_db)) -> GraphStore:
    """Dependency to get GraphStore instance."""
    from knowledge_base.persistence.v1.graph_store import GraphStore

    return GraphStore(db)


@router.get("/{graph_id}:summary", response_model=APIResponse[GraphSummaryResponse])
async def get_graph_summary(
    graph_id: UUID = Path(..., description="Graph ID"),
    level: int = Query(0, ge=0, le=5, description="Community hierarchy level"),
    min_community_size: int = Query(
        3, ge=1, description="Minimum entities per community"
    ),
    include_metrics: bool = Query(False, description="Include centrality metrics"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    store: GraphStore = Depends(get_graph_store),
):
    """
    GET /api/v1/graphs/{graph_id}:summary

    Returns community-level graph overview for Sigma.js initial view.
    Uses Map-Reduce style aggregation to efficiently calculate inter-community edges.

    AIP-136 Custom Method pattern for operations that don't fit standard CRUD.
    """
    try:
        communities, aggregated_edges = await store.get_community_topology(
            level=level, min_community_size=min_community_size, domain=domain
        )

        if not communities:
            return APIResponse(
                success=True,
                data=GraphSummaryResponse(
                    nodes=[], edges=[], total_nodes=0, total_edges=0, stats={}
                ),
                error=None,
                metadata={"graph_id": str(graph_id)},
            )

        # Convert communities to GraphNode format
        nodes = []
        for community in communities:
            node_attributes = {
                "label": str(community.name)
                if community.name
                else f"Community {community.id}",
                "x": 0,  # Will be calculated by frontend or from metadata if available
                "y": 0,
                "size": community.entity_count or 1,
                "color": community.metadata.get("color", "#4CAF50")
                if community.metadata
                else "#4CAF50",
                "community_id": str(community.id),
                "level": level,
                "entity_count": community.entity_count,
                "summary": str(community.summary) if community.summary else "",
                "domain": str(community.domain) if community.domain else "",
            }

            # If metadata has pre-calculated coordinates, use them
            if community.metadata and "coordinates" in community.metadata:
                coords = community.metadata["coordinates"]
                node_attributes["x"] = coords.get("x", 0)
                node_attributes["y"] = coords.get("y", 0)

            nodes.append(
                GraphNode(
                    key=str(community.id),
                    label=node_attributes["label"],
                    attributes=node_attributes,
                )
            )

        # Convert aggregated edges to GraphEdge format
        edges = []
        for agg in aggregated_edges:
            edge_attributes = {
                "weight": agg.weight,
                "confidence": agg.avg_confidence,
                "label": f"{agg.weight} connections",
                "edge_types": agg.edge_types,
                "color": "rgba(0,0,0,0.3)",
                "size": min(agg.weight / 10, 5),  # Scale edge size
            }

            edges.append(
                GraphEdge(
                    key=f"{agg.source}-{agg.target}",
                    source=str(agg.source),
                    target=str(agg.target),
                    attributes=edge_attributes,
                )
            )

        # Calculate graph statistics
        if include_metrics:
            stats = await store.get_graph_statistics()
        else:
            stats = {
                "community_count": len(communities),
                "inter_community_edges": len(edges),
            }

        response = GraphSummaryResponse(
            nodes=nodes,
            edges=edges,
            total_nodes=len(nodes),
            total_edges=len(edges),
            stats=stats,
        )

        return APIResponse(
            success=True,
            data=response,
            error=None,
            metadata={
                "graph_id": str(graph_id),
                "level": level,
                "community_count": len(communities),
            },
        )

    except Exception as e:
        # Error will be handled by exception handler middleware
        raise


@router.get(
    "/{graph_id}/nodes/{node_id}:neighborhood",
    response_model=APIResponse[NeighborhoodResponse],
)
async def get_neighborhood(
    graph_id: UUID = Path(..., description="Graph ID"),
    node_id: UUID = Path(..., description="Center node ID"),
    depth: int = Query(1, ge=1, le=3, description="Traversal depth"),
    direction: str = Query(
        "bidirectional",
        pattern="^(outgoing|incoming|bidirectional)$",
        description="Edge direction to follow",
    ),
    min_confidence: float = Query(
        0.7, ge=0.0, le=1.0, description="Minimum confidence"
    ),
    max_nodes: int = Query(1000, ge=1, le=5000, description="Maximum nodes to return"),
    node_types: Optional[List[str]] = Query(None, description="Filter by entity types"),
    edge_types: Optional[List[str]] = Query(None, description="Filter by edge types"),
    store: GraphStore = Depends(get_graph_store),
):
    """
    GET /api/v1/graphs/{graph_id}/nodes/{node_id}:neighborhood

    Expand entity neighborhood for Sigma.js drill-down interactions.
    Returns immediate neighbors and connecting edges with confidence filtering.

    AIP-136 Custom Method pattern for complex graph operations.
    """
    try:
        # Convert direction string to enum
        direction_enum = GraphTraversalDirection(direction)

        center_entity, neighborhood = await store.get_entity_neighborhood(
            entity_id=node_id,
            depth=depth,
            direction=direction_enum,
            min_confidence=min_confidence,
            max_nodes=max_nodes,
            node_types=node_types,
            edge_types=edge_types,
        )

        if not neighborhood:
            # Return just the center node if no neighbors
            center_node = GraphNode(
                key=str(center_entity.id),
                label=center_entity.name or str(center_entity.id),
                attributes={
                    "x": 0,
                    "y": 0,
                    "size": 10,
                    "color": "#2196F3",
                    "entity_type": center_entity.entity_type,
                    "confidence": center_entity.confidence,
                    "community_id": str(center_entity.community_id)
                    if center_entity.community_id
                    else None,
                },
            )

            return APIResponse(
                success=True,
                data=NeighborhoodResponse(
                    center_node=center_node,
                    nodes=[center_node],
                    edges=[],
                    expanded_count=0,
                    max_confidence=0.0,
                    min_confidence=0.0,
                ),
                error=None,
                metadata={"graph_id": str(graph_id), "node_id": str(node_id)},
            )

        # Track nodes and edges for response
        nodes_by_id = {}
        edges = []
        confidences = []

        # Add center node first
        center_node = GraphNode(
            key=str(center_entity.id),
            label=center_entity.name or str(center_entity.id),
            attributes={
                "x": 0,  # Center at (0, 0)
                "y": 0,
                "size": 15,  # Larger for center
                "color": "#2196F3",  # Blue for center
                "entity_type": center_entity.entity_type,
                "confidence": center_entity.confidence,
                "community_id": str(center_entity.community_id)
                if center_entity.community_id
                else None,
            },
        )
        nodes_by_id[str(center_entity.id)] = center_node

        # Process neighbors and edges
        for neighbor_entity, edge in neighborhood:
            neighbor_id = str(neighbor_entity.id)

            # Add neighbor node if not already added
            if neighbor_id not in nodes_by_id:
                neighbor_node = GraphNode(
                    key=neighbor_id,
                    label=neighbor_entity.name or neighbor_id,
                    attributes={
                        "x": 0,  # Will be positioned by frontend
                        "y": 0,
                        "size": 10,
                        "color": "#757575",  # Gray for neighbors
                        "entity_type": neighbor_entity.entity_type,
                        "confidence": neighbor_entity.confidence,
                        "community_id": str(neighbor_entity.community_id)
                        if neighbor_entity.community_id
                        else None,
                    },
                )
                nodes_by_id[neighbor_id] = neighbor_node

            # Add edge
            edge_id = str(edge.id)
            edge_attributes = {
                "weight": edge.weight or 1,
                "confidence": edge.confidence or 0.5,
                "label": edge.edge_type,
                "type": edge.edge_type,
                "color": "rgba(0,0,0,0.4)",
                "line_style": edge.properties.get("line_style", "solid")
                if edge.properties
                else "solid",
            }

            graph_edge = GraphEdge(
                key=edge_id,
                source=str(edge.source_id),
                target=str(edge.target_id),
                attributes=edge_attributes,
            )
            edges.append(graph_edge)

            if edge.confidence:
                confidences.append(edge.confidence)

        # Calculate confidence metrics
        max_confidence = max(confidences) if confidences else 0.0
        min_confidence = min(confidences) if confidences else 0.0

        response = NeighborhoodResponse(
            center_node=center_node,
            nodes=list(nodes_by_id.values()),
            edges=edges,
            expanded_count=len(neighborhood),
            max_confidence=max_confidence,
            min_confidence=min_confidence,
        )

        return APIResponse(
            success=True,
            data=response,
            error=None,
            metadata={
                "graph_id": str(graph_id),
                "node_id": str(node_id),
                "depth": depth,
                "direction": direction,
                "neighbor_count": len(neighborhood),
            },
        )

    except Exception as e:
        raise


@router.post("/{graph_id}:findPath", response_model=APIResponse[PathResponse])
async def find_path(
    graph_id: UUID = Path(..., description="Graph ID"),
    request: PathFindingRequest = Body(...),
    store: GraphStore = Depends(get_graph_store),
):
    """
    POST /api/v1/graphs/{graph_id}:findPath

    Find paths between two nodes using various algorithms.
    Server-side path finding avoids moving large datasets to client.

    AIP-136 Custom Method for complex graph analytics.
    """
    try:
        # Extract source and target node IDs from paths if provided as full resource names
        import re

        def extract_node_id(node_path: str) -> str:
            """Extract UUID from resource path like 'graphs/123/nodes/456'."""
            match = re.search(r"nodes/([0-9a-f-]+)$", node_path, re.I)
            if match:
                return match.group(1)
            return node_path

        source_id = UUID(extract_node_id(request.source))
        target_id = UUID(extract_node_id(request.target))

        # This is a simplified path finding implementation
        # In production, this would integrate with NetworkX or a graph database

        if request.algorithm == "shortest_path":
            # Simple breadth-first search for shortest path
            paths, confidence_scores = await _find_shortest_paths(
                store,
                source_id,
                target_id,
                request.max_hops,
                request.min_confidence,
                request.edge_types,
            )
        elif request.algorithm == "all_simple_paths":
            paths, confidence_scores = await _find_all_simple_paths(
                store,
                source_id,
                target_id,
                request.max_hops,
                request.min_confidence,
                request.edge_types,
            )
        else:  # most_confident
            paths, confidence_scores = await _find_most_confident_paths(
                store,
                source_id,
                target_id,
                request.max_hops,
                request.min_confidence,
                request.edge_types,
            )

        path_response = PathResponse(
            source=request.source,
            target=request.target,
            paths=paths,
            confidence_scores=confidence_scores,
            total_paths=len(paths),
            algorithm=request.algorithm,
        )

        return APIResponse(
            success=True,
            data=path_response,
            error=None,
            metadata={
                "graph_id": str(graph_id),
                "calculation_time_ms": 0,  # Add timing if needed
            },
        )

    except Exception as e:
        raise


@router.get("/{graph_id}:export", response_model=APIResponse[GraphologyExport])
async def export_graph(
    graph_id: UUID = Path(..., description="Graph ID"),
    format: str = Query("graphology", pattern="^(graphology|gexf|graphml)$"),
    include_layout: bool = Query(
        False, description="Include pre-calculated layout coordinates"
    ),
    store: GraphStore = Depends(get_graph_store),
):
    """
    GET /api/v1/graphs/{graph_id}:export

    Export graph in Graphology JSON format for Sigma.js compatibility.
    Can include pre-calculated layout coordinates for large graphs.

    AIP-136 Custom Method for data export operations.
    """
    try:
        # Get graph statistics
        stats = await store.get_graph_statistics()

        # For now, return a template response
        # Full implementation would fetch all entities and edges
        export = GraphologyExport(
            attributes={
                "name": f"Graph {graph_id}",
                "entity_count": stats.get("entity_count", 0),
                "edge_count": stats.get("edge_count", 0),
                "generated_at": datetime.utcnow().isoformat(),
            },
            nodes=[],
            edges=[],
        )

        return APIResponse(
            success=True,
            data=export,
            error=None,
            metadata={
                "graph_id": str(graph_id),
                "format": format,
                "include_layout": include_layout,
            },
        )

    except Exception as e:
        raise


# Helper functions for path finding
logger = logging.getLogger(__name__)


async def _find_shortest_paths(
    store: GraphStore,
    source_id: UUID,
    target_id: UUID,
    max_hops: int,
    min_confidence: float,
    edge_types: Optional[List[str]],
) -> Tuple[List[List[str]], List[float]]:
    """Find shortest paths between source and target using igraph."""
    import igraph

    # Get graph data from store
    nodes_result = await store.db.execute(
        select(Entity).where(Entity.community_id.isnot(None))
    )
    nodes = nodes_result.scalars().all()

    if not nodes:
        return [], []

    # Build igraph
    G = igraph.Graph(directed=True)

    # Add vertices
    node_ids = [str(node.id) for node in nodes]
    G.add_vertices(node_ids)

    # Add vertex attributes
    G.vs["name"] = [node.name for node in nodes]
    G.vs["type"] = [node.entity_type for node in nodes]

    # Get edges
    edges_result = await store.db.execute(
        select(Edge).where(
            Edge.confidence >= min_confidence,
            Edge.source_id.in_([UUID(nid) for nid in node_ids]),
            Edge.target_id.in_([UUID(nid) for nid in node_ids]),
        )
    )
    edges = edges_result.scalars().all()

    # Filter by edge types if specified
    if edge_types:
        edges = [e for e in edges if e.edge_type in edge_types]

    # Add edges with weights (higher confidence = lower weight)
    edge_list = []
    edge_weights = []
    for edge in edges:
        if str(edge.source_id) in node_ids and str(edge.target_id) in node_ids:
            source_idx = node_ids.index(str(edge.source_id))
            target_idx = node_ids.index(str(edge.target_id))
            edge_list.append((source_idx, target_idx))
            edge_weights.append(1.0 - edge.confidence)

    G.add_edges(edge_list)
    G.es["weight"] = edge_weights
    G.es["confidence"] = [1.0 - w for w in edge_weights]

    # Find shortest paths
    try:
        paths = G.get_shortest_paths(
            str(source_id), str(target_id), weights="weight", mode="OUT"
        )

        # Convert to node ID strings and calculate path confidence
        result_paths = []
        confidences = []

        for path in paths:
            if 0 < len(path) <= max_hops + 1:
                path_ids = [node_ids[i] for i in path]
                result_paths.append(path_ids)

                # Calculate average confidence for this path
                if len(path) > 1:
                    path_edges = []
                    for i in range(len(path) - 1):
                        e = G.es.select(_source=path[i], _target=path[i + 1])
                        if e:
                            path_edges.append(e[0])
                    if path_edges:
                        avg_confidence = sum(e["confidence"] for e in path_edges) / len(
                            path_edges
                        )
                    else:
                        avg_confidence = 1.0
                else:
                    avg_confidence = 1.0
                confidences.append(avg_confidence)

        return result_paths, confidences

    except Exception as e:
        logger.error(f"Shortest path finding failed: {e}")
        return [], []


async def _find_all_simple_paths(
    store: GraphStore,
    source_id: UUID,
    target_id: UUID,
    max_hops: int,
    min_confidence: float,
    edge_types: Optional[List[str]],
) -> Tuple[List[List[str]], List[float]]:
    """Find all simple paths between source and target using igraph."""
    import igraph

    # Get graph data from store
    nodes_result = await store.db.execute(
        select(Entity).where(Entity.community_id.isnot(None))
    )
    nodes = nodes_result.scalars().all()

    if not nodes:
        return [], []

    # Build igraph
    G = igraph.Graph(directed=True)

    # Add vertices
    node_ids = [str(node.id) for node in nodes]
    G.add_vertices(node_ids)

    # Add vertex attributes
    G.vs["name"] = [node.name for node in nodes]
    G.vs["type"] = [node.entity_type for node in nodes]

    # Get edges
    edges_result = await store.db.execute(
        select(Edge).where(
            Edge.confidence >= min_confidence,
            Edge.source_id.in_([UUID(nid) for nid in node_ids]),
            Edge.target_id.in_([UUID(nid) for nid in node_ids]),
        )
    )
    edges = edges_result.scalars().all()

    # Filter by edge types if specified
    if edge_types:
        edges = [e for e in edges if e.edge_type in edge_types]

    # Add edges with confidence as attribute
    edge_list = []
    edge_confidences = []
    for edge in edges:
        if str(edge.source_id) in node_ids and str(edge.target_id) in node_ids:
            source_idx = node_ids.index(str(edge.source_id))
            target_idx = node_ids.index(str(edge.target_id))
            edge_list.append((source_idx, target_idx))
            edge_confidences.append(edge.confidence)

    G.add_edges(edge_list)
    G.es["confidence"] = edge_confidences

    # Find all simple paths
    try:
        paths = G.get_all_simple_paths(str(source_id), str(target_id))

        # Convert to node ID strings and calculate path confidence
        result_paths = []
        confidences = []

        for path in paths:
            if len(path) > 0:
                path_ids = [node_ids[i] for i in path]
                result_paths.append(path_ids)

                # Calculate average confidence for this path
                if len(path) > 1:
                    path_edges = []
                    for i in range(len(path) - 1):
                        e = G.es.select(_source=path[i], _target=path[i + 1])
                        if e:
                            path_edges.append(e[0])
                    if path_edges:
                        avg_confidence = sum(e["confidence"] for e in path_edges) / len(
                            path_edges
                        )
                    else:
                        avg_confidence = 1.0
                else:
                    avg_confidence = 1.0
                confidences.append(avg_confidence)

        return result_paths, confidences

    except Exception as e:
        logger.error(f"All simple paths finding failed: {e}")
        return [], []


async def _find_most_confident_paths(
    store: GraphStore,
    source_id: UUID,
    target_id: UUID,
    max_hops: int,
    min_confidence: float,
    edge_types: Optional[List[str]],
) -> Tuple[List[List[str]], List[float]]:
    """Find paths with highest confidence scores using igraph."""
    import igraph

    # Get graph data from store
    nodes_result = await store.db.execute(
        select(Entity).where(Entity.community_id.isnot(None))
    )
    nodes = nodes_result.scalars().all()

    if not nodes:
        return [], []

    # Build igraph
    G = igraph.Graph(directed=True)

    # Add vertices
    node_ids = [str(node.id) for node in nodes]
    G.add_vertices(node_ids)

    # Add vertex attributes
    G.vs["name"] = [node.name for node in nodes]
    G.vs["type"] = [node.entity_type for node in nodes]

    # Get edges
    edges_result = await store.db.execute(
        select(Edge).where(
            Edge.confidence >= min_confidence,
            Edge.source_id.in_([UUID(nid) for nid in node_ids]),
            Edge.target_id.in_([UUID(nid) for nid in node_ids]),
        )
    )
    edges = edges_result.scalars().all()

    # Filter by edge types if specified
    if edge_types:
        edges = [e for e in edges if e.edge_type in edge_types]

    # Add edges with weights (higher confidence = lower weight)
    edge_list = []
    edge_weights = []
    edge_confidences = []
    for edge in edges:
        if str(edge.source_id) in node_ids and str(edge.target_id) in node_ids:
            source_idx = node_ids.index(str(edge.source_id))
            target_idx = node_ids.index(str(edge.target_id))
            edge_list.append((source_idx, target_idx))
            # Use inverse of confidence as weight (higher confidence = lower weight)
            weight = 1.0 - edge.confidence
            edge_weights.append(weight)
            edge_confidences.append(edge.confidence)

    G.add_edges(edge_list)
    G.es["weight"] = edge_weights
    G.es["confidence"] = edge_confidences

    # Find shortest paths using confidence-based weights
    try:
        paths = G.get_shortest_paths(
            str(source_id), str(target_id), weights="weight", mode="OUT"
        )

        # Get all simple paths as well to find alternative confident paths (without cutoff parameter)
        all_paths = G.get_all_simple_paths(str(source_id), str(target_id))

        # Combine and rank by confidence
        combined_paths = []
        path_confidences = []

        # Add shortest paths
        for path in paths:
            if 0 < len(path) <= max_hops + 1:
                combined_paths.append(path)

        # Add all simple paths
        for path in all_paths:
            if 0 < len(path) <= max_hops + 1 and path not in combined_paths:
                combined_paths.append(path)

        # Calculate confidence for each path and sort
        scored_paths = []
        for path in combined_paths:
            path_ids = [node_ids[i] for i in path]

            # Calculate average confidence for this path
            if len(path) > 1:
                path_edges = []
                for i in range(len(path) - 1):
                    e = G.es.select(_source=path[i], _target=path[i + 1])
                    if e:
                        path_edges.append(e[0])
                if path_edges:
                    min_edge_conf = min(e["confidence"] for e in path_edges)
                    avg_confidence = sum(e["confidence"] for e in path_edges) / len(
                        path_edges
                    )
                    # Combine min and average confidence
                    path_confidence = 0.7 * min_edge_conf + 0.3 * avg_confidence
                else:
                    path_confidence = 1.0
            else:
                path_confidence = 1.0

            scored_paths.append((path_ids, path_confidence))

        # Sort by confidence (descending) and return top paths
        scored_paths.sort(key=lambda x: x[1], reverse=True)

        result_paths = [sp[0] for sp in scored_paths]
        confidences = [sp[1] for sp in scored_paths]

        return result_paths, confidences

    except Exception as e:
        logger.error(f"Most confident paths finding failed: {e}")
        return [], []
