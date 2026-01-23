"""Hierarchical Leiden clustering service."""

import asyncio
from uuid import UUID

import igraph as ig
import leidenalg as la
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from knowledge_base.persistence.v1.schema import Community, Edge, Entity


class ClusteringConfig(BaseSettings):
    """Clustering configuration."""

    model_config = SettingsConfigDict()

    resolution_macro: float = 0.8
    resolution_micro: float = 1.2
    iterations: int = 10
    min_community_size: int = 3


class ClusteringResult(BaseModel):
    """Clustering result."""

    partition: dict[UUID, int] = Field(
        ..., description="Entity ID to community mapping"
    )
    communities: dict[int, list[UUID]] = Field(
        ..., description="Community ID to entity IDs"
    )
    modularity: float = Field(..., description="Partition modularity score")
    community_count: int = Field(..., description="Number of communities")


class CommunityHierarchy(BaseModel):
    """Community hierarchy."""

    model_config = {"arbitrary_types_allowed": True}

    community_ids: list[UUID] = Field(default_factory=list, description="Community IDs")
    communities: dict[UUID, Community] = Field(
        default_factory=dict,
        description="Community objects",
    )
    hierarchy: dict[UUID, UUID | None] = Field(
        default_factory=dict,
        description="Parent mapping",
    )


class ClusteringService:
    """Service for hierarchical Leiden clustering."""

    def __init__(self, config: ClusteringConfig | None = None) -> None:
        """Initialize clustering service.

        Args:
            config: Clustering configuration.
        """
        self._config = config or ClusteringConfig()

    async def cluster_entities(
        self,
        entities: list[Entity],
        edges: list[Edge],
        resolution: float | None = None,
    ) -> ClusteringResult:
        """Cluster entities using Leiden algorithm.

        Args:
            entities: List of entities to cluster.
            edges: List of edges between entities.
            resolution: Resolution parameter. If None, uses config default.

        Returns:
            Clustering result.
        """
        resolution = resolution or self._config.resolution_macro

        return await asyncio.to_thread(
            self._cluster_sync,
            entities,
            edges,
            resolution,
        )

    def _cluster_sync(
        self,
        entities: list[Entity],
        edges: list[Edge],
        resolution: float,
    ) -> ClusteringResult:
        """Synchronous clustering using igraph and leidenalg.

        Args:
            entities: List of entities.
            edges: List of edges.
            resolution: Resolution parameter.

        Returns:
            Clustering result.
        """
        entity_id_to_index = {entity.id: i for i, entity in enumerate(entities)}

        g = ig.Graph()
        g.add_vertices(len(entities))

        edge_tuples = []
        for edge in edges:
            if (
                edge.source_id in entity_id_to_index
                and edge.target_id in entity_id_to_index
            ):
                source_idx = entity_id_to_index[edge.source_id]
                target_idx = entity_id_to_index[edge.target_id]
                edge_tuples.append((source_idx, target_idx))

        if edge_tuples:
            g.add_edges(edge_tuples)
            g.es["weight"] = [1.0] * len(edge_tuples)
        else:
            return ClusteringResult(
                partition={},
                communities={},
                modularity=0.0,
                community_count=0,
            )

        partition = la.find_partition(
            g,
            la.ModularityVertexPartition,
            n_iterations=self._config.iterations,
        )

        modularity = partition.quality()

        partition_dict: dict[UUID, int] = {}
        communities_dict: dict[int, list[UUID]] = {}

        for i, entity in enumerate(entities):
            community_id = partition.membership[i]
            partition_dict[entity.id] = community_id

            if community_id not in communities_dict:
                communities_dict[community_id] = []
            communities_dict[community_id].append(entity.id)

        communities_dict = {
            cid: entities_
            for cid, entities_ in communities_dict.items()
            if len(entities_) >= self._config.min_community_size
        }

        return ClusteringResult(
            partition=partition_dict,
            communities=communities_dict,
            modularity=modularity,
            community_count=len(communities_dict),
        )

    async def build_hierarchy(
        self,
        entities: list[Entity],
        edges: list[Edge],
    ) -> CommunityHierarchy:
        """Build hierarchical community structure.

        Args:
            entities: List of entities.
            edges: List of edges.

        Returns:
            Community hierarchy.
        """
        macro_result = await self.cluster_entities(
            entities, edges, self._config.resolution_macro
        )

        communities: dict[UUID, Community] = {}
        entity_to_community: dict[UUID, UUID] = {}

        from uuid import uuid4

        for comm_id, entity_ids in macro_result.communities.items():
            comm_uuid = uuid4()

            entity_names = ", ".join(
                next(e.name for e in entities if e.id == eid) for eid in entity_ids
            )

            communities[comm_uuid] = Community(
                id=comm_uuid,
                name=f"Macro Community {comm_id}: {entity_names[:50]}...",
                level=0,
                resolution=self._config.resolution_macro,
                entity_count=len(entity_ids),
            )

            for eid in entity_ids:
                entity_to_community[eid] = comm_uuid

        hierarchy: dict[UUID, UUID | None] = {cid: None for cid in communities}

        if len(macro_result.communities) > 3:
            micro_result = await self.cluster_entities(
                entities,
                edges,
                self._config.resolution_micro,
            )

            for comm_id, entity_ids in micro_result.communities.items():
                comm_uuid = uuid4()

                entity_names = ", ".join(
                    next(e.name for e in entities if e.id == eid) for eid in entity_ids
                )

                communities[comm_uuid] = Community(
                    id=comm_uuid,
                    name=f"Micro Community {comm_id}: {entity_names[:50]}...",
                    level=1,
                    resolution=self._config.resolution_micro,
                    entity_count=len(entity_ids),
                )

                parent_community_id = entity_to_community.get(entity_ids[0])
                if parent_community_id:
                    hierarchy[comm_uuid] = parent_community_id

        return CommunityHierarchy(
            community_ids=list(communities.keys()),
            communities=communities,
            hierarchy=hierarchy,
        )

    async def incremental_update(
        self,
        existing_result: ClusteringResult,
        new_entities: list[Entity],
        new_edges: list[Edge],
        all_entities: list[Entity],
        all_edges: list[Edge],
    ) -> ClusteringResult:
        """Update clustering incrementally with new entities.

        Args:
            existing_result: Existing clustering result.
            new_entities: New entities to add.
            new_edges: New edges to add.
            all_entities: All entities (existing + new).
            all_edges: All edges (existing + new).

        Returns:
            Updated clustering result.
        """
        return await self.cluster_entities(all_entities, all_edges)
