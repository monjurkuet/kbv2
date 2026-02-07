"""Entity pipeline service for entity extraction and resolution."""

import logging
import re
from datetime import datetime
from typing import List, Optional, Tuple
from urllib.parse import quote
from uuid import UUID, uuid4

from knowledge_base.orchestration.base_service import BaseService
from knowledge_base.config.constants import (
    ENTITY_SIMILARITY_THRESHOLD,
    MIN_ENTITY_CONFIDENCE,
)
from knowledge_base.persistence.v1.schema import (
    Document,
    Entity,
    Edge,
    Chunk,
    ChunkEntity,
)
from knowledge_base.intelligence.v1.multi_agent_extractor import (
    EntityExtractionManager,
    ExtractionQualityScore,
    ExtractedEntity,
)
from knowledge_base.intelligence.v1.resolution_agent import (
    EntityResolution,
    ResolutionAgent,
)
from knowledge_base.intelligence.v1.entity_typing_service import (
    EntityTyper,
    TypedEntity,
)
from knowledge_base.intelligence.v1.clustering_service import (
    ClusteringService,
)
from knowledge_base.intelligence.v1.hallucination_detector import (
    HallucinationDetector,
    EntityVerification,
    RiskLevel,
)
from knowledge_base.ingestion.v1.gleaning_service import (
    ExtractionResult,
    GleaningService,
)
from knowledge_base.persistence.v1.graph_store import GraphStore
from knowledge_base.orchestration.quality_assurance_service import (
    QualityAssuranceService,
)
from knowledge_base.summaries.community_summaries import (
    CommunitySummarizer,
    MultiLevelSummary,
)
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSONB


class EntityPipelineService(BaseService):
    """Service for entity extraction, resolution, and clustering."""

    def __init__(self):
        super().__init__()
        self._multi_agent_extractor: Optional[EntityExtractionManager] = None
        self._entity_resolver: Optional[ResolutionAgent] = None
        self._entity_typer: Optional[EntityTyper] = None
        self._clustering_service: Optional[ClusteringService] = None
        self._gleaning_service: Optional[GleaningService] = None
        self._hallucination_detector: Optional[HallucinationDetector] = None
        self._community_summarizer: Optional[CommunitySummarizer] = None
        self._vector_store = None
        self._gateway = None

    async def initialize(
        self,
        vector_store=None,
        gateway=None,
    ) -> None:
        """Initialize the service."""
        self._vector_store = vector_store
        self._gateway = gateway
        self._logger.info("EntityPipelineService initialized")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        self._logger.info("EntityPipelineService shutdown")

    def set_extractors(
        self,
        multi_agent_extractor: Optional[EntityExtractionManager],
        entity_resolver: Optional[ResolutionAgent],
        entity_typer: Optional[EntityTyper],
        clustering_service: Optional[ClusteringService],
        gleaning_service: Optional[GleaningService],
        hallucination_detector: Optional[HallucinationDetector],
    ):
        """Set the extraction components."""
        self._multi_agent_extractor = multi_agent_extractor
        self._entity_resolver = entity_resolver
        self._entity_typer = entity_typer
        self._clustering_service = clustering_service
        self._gleaning_service = gleaning_service
        self._hallucination_detector = hallucination_detector

    async def extract_entities(
        self,
        document: Document,
        chunks: List[Chunk],
        domain: str,
        use_multi_agent: bool = True,
        recommendation=None,
    ) -> Tuple[
        List[Entity],
        List[Edge],
        Optional[ExtractionQualityScore],
        List[EntityVerification],
    ]:
        """Extract entities and relationships from a document.

        Args:
            document: The document to extract from
            chunks: List of chunks
            domain: Document domain
            use_multi_agent: If True, try multi-agent first, fallback to gleaning
            recommendation: Adaptive ingestion recommendation (optional)

        Returns:
            Tuple of (entities, edges, quality_score, verifications)
        """
        quality_score = None
        verifications = []

        # Determine extraction approach based on recommendation
        should_use_multi_agent = use_multi_agent
        if recommendation and not recommendation.use_multi_agent:
            should_use_multi_agent = False
            self._logger.info(
                f"Adaptive analysis recommends gleaning approach: "
                f"{recommendation.justification}"
            )

        if should_use_multi_agent and self._multi_agent_extractor:
            try:
                (
                    entities,
                    edges,
                    quality_score,
                    extracted_verifications,
                ) = await self._extract_entities_multi_agent(
                    document=document,
                    chunks=chunks,
                    domain=domain,
                    recommendation=recommendation,
                )
                verifications = extracted_verifications
                return entities, edges, quality_score, verifications
            except Exception as e:
                self._logger.warning(
                    f"Multi-agent extraction failed, falling back to gleaning: {e}"
                )

        # Fallback to gleaning
        if self._gleaning_service:
            entities, edges = await self._extract_entities_gleaning(chunks)
            return entities, edges, None, verifications

        return [], [], None, verifications

    async def _extract_entities_multi_agent(
        self,
        document: Document,
        chunks: List[Chunk],
        domain: str,
        recommendation=None,
    ) -> Tuple[
        List[Entity],
        List[Edge],
        Optional[ExtractionQualityScore],
        List[EntityVerification],
    ]:
        """Extract entities using multi-agent system with quality verification."""
        if not self._multi_agent_extractor:
            raise ValueError("Multi-agent extractor not set")

        async with self._vector_store.get_session() as session:
            graph_store = GraphStore(session)

            # Create extraction manager with config overrides if recommendation provided
            config_kwargs = {}
            if recommendation:
                config_kwargs = {
                    "enhancement_max_iterations": recommendation.max_enhancement_iterations,
                    "confidence_threshold": recommendation.confidence_threshold,
                }
                self._logger.info(
                    f"Using adaptive config: max_iterations={recommendation.max_enhancement_iterations}, "
                    f"confidence_threshold={recommendation.confidence_threshold}"
                )

            extraction_manager = EntityExtractionManager(
                gateway=self._gateway,
                graph_store=graph_store,
                vector_store=self._vector_store,
                **config_kwargs,
            )

            (
                extracted_entities,
                quality_score,
            ) = await extraction_manager.extract_with_evaluation(
                list(chunks), document.id
            )

            entities, edges = self._convert_extraction_to_entities(
                extracted_entities, chunks
            )

            verifications = []
            if self._hallucination_detector and entities:
                for entity in entities:
                    verification = await self._hallucination_detector.verify_entity(
                        entity_name=entity.name,
                        entity_type=entity.entity_type or "UNKNOWN",
                        attributes=entity.properties or {},
                        context=entity.source_text or "",
                    )
                    verifications.append(verification)

            return entities, edges, quality_score, verifications

    async def _extract_entities_gleaning(
        self,
        chunks: List[Chunk],
    ) -> Tuple[List[Entity], List[Edge]]:
        """Extract entities using gleaning service."""
        if not self._gleaning_service:
            raise ValueError("Gleaning service not set")

        entities = []
        edges = []

        for chunk in chunks:
            chunk_text = str(chunk.text) if hasattr(chunk.text, "value") else chunk.text
            extraction = await self._gleaning_service.extract(chunk_text)

            chunk_entities = self._create_entities_from_extraction(extraction, chunk.id)
            entities.extend(chunk_entities)

            chunk_edges = self._create_edges_from_extraction(extraction, chunk_entities)
            edges.extend(chunk_edges)

        return entities, edges

    def _create_entities_from_extraction(
        self,
        extraction: ExtractionResult,
        chunk_id: UUID,
    ) -> List[Entity]:
        """Create entities from extraction result."""
        entities = []
        for entity in extraction.entities:
            # Generate a stable URI for the entity
            normalized_name = re.sub(
                r"[^\w\s-]", "", entity.name.lower().replace(" ", "_")
            )
            uri = f"entity:{quote(normalized_name)}"

            db_entity = Entity(
                id=uuid4(),
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                properties=entity.properties,
                confidence=entity.confidence,
                uri=uri,
            )
            entities.append(db_entity)

        return entities

    def _create_edges_from_extraction(
        self,
        extraction: ExtractionResult,
        entities: List[Entity],
    ) -> List[Edge]:
        """Create edges from extraction result."""
        entity_map = {e.name: e.id for e in entities}

        edges = []
        for edge_data in extraction.edges:
            if edge_data.source in entity_map and edge_data.target in entity_map:
                edge = Edge(
                    id=uuid4(),
                    source_id=entity_map[edge_data.source],
                    target_id=entity_map[edge_data.target],
                    edge_type=edge_data.edge_type,
                    properties=edge_data.properties,
                    confidence=edge_data.confidence,
                    provenance=edge_data.provenance,
                    source_text=edge_data.source_text,
                    temporal_validity_start=edge_data.temporal_validity_start,
                    temporal_validity_end=edge_data.temporal_validity_end,
                )
                edges.append(edge)

        return edges

    def _convert_extraction_to_entities(
        self,
        extracted_entities: List[ExtractedEntity],
        chunks: List[Chunk],
    ) -> Tuple[List[Entity], List[Edge]]:
        """Convert multi-agent extraction result to persistence entities."""
        entities = []
        edges = []

        entity_map = {}

        for extracted in extracted_entities:
            normalized_name = re.sub(
                r"[^\w\s-]", "", extracted.name.lower().replace(" ", "_")
            )
            uri = f"entity:{quote(normalized_name)}"

            entity = Entity(
                id=extracted.id if extracted.id else uuid4(),
                name=extracted.name,
                entity_type=extracted.entity_type,
                description=extracted.description,
                properties=extracted.properties,
                confidence=extracted.confidence,
                uri=uri,
                source_text=extracted.source_text,
                domain=None,
            )
            entities.append(entity)
            entity_map[extracted.id] = entity

        for extracted in extracted_entities:
            for linked_id in extracted.linked_entities:
                if linked_id in entity_map:
                    source = entity_map.get(extracted.id)
                    target = entity_map.get(linked_id)
                    if source and target and source.id != target.id:
                        edge = Edge(
                            id=uuid4(),
                            source_id=source.id,
                            target_id=target.id,
                            edge_type="RELATED_TO",
                            properties={},
                            confidence=extracted.confidence,
                            provenance="multi_agent_extraction",
                            source_text=extracted.source_text,
                        )
                        edges.append(edge)

        return entities, edges

    async def resolve_entities(
        self,
        document: Document,
        entities: List[Entity],
        session,
    ) -> List[EntityResolution]:
        """Resolve duplicate entities.

        Args:
            document: The document
            entities: List of entities to resolve
            session: Database session

        Returns:
            List of entity resolutions
        """
        if not self._entity_resolver:
            return []

        resolutions = []

        for entity in entities:
            # Check for duplicates in the database
            embedding = entity.embedding
            if embedding is not None:
                embedding = (
                    embedding.tolist() if hasattr(embedding, "tolist") else embedding
                )
            else:
                embedding = []

            similar = await self._vector_store.search_similar_entities(
                embedding,
                limit=5,
                similarity_threshold=ENTITY_SIMILARITY_THRESHOLD,
            )

            # Extract candidate IDs from similarity results
            candidate_ids = []
            for s in similar:
                s_id_str = str(s["id"])
                if s_id_str != str(entity.id):
                    try:
                        candidate_ids.append(UUID(s_id_str))
                    except (ValueError, TypeError):
                        continue

            if candidate_ids:
                # Fetch the actual entity objects for global resolution
                candidate_result = await session.execute(
                    select(Entity).where(Entity.id.in_(candidate_ids))
                )
                candidates = candidate_result.scalars().all()
            else:
                candidates = []

            if candidates:
                # Get chunk context for grounding
                chunk_text = ""
                chunk_result = await session.execute(
                    select(Chunk)
                    .join(ChunkEntity, ChunkEntity.chunk_id == Chunk.id)
                    .where(ChunkEntity.entity_id == entity.id)
                    .limit(1)
                )
                chunk = chunk_result.scalar_one_or_none()
                if chunk:
                    chunk_text = (
                        str(chunk.text) if hasattr(chunk.text, "value") else chunk.text
                    )

                resolution = await self._entity_resolver.resolve_entity(
                    entity,
                    candidates,
                    chunk_text,
                )

                if resolution.merged_entity_ids:
                    resolutions.append(resolution)

        return resolutions

    async def merge_entities(
        self,
        resolution: EntityResolution,
        session,
    ) -> None:
        """Merge entities based on resolution.

        Args:
            resolution: Entity resolution result.
            session: Database session
        """
        target_id = resolution.entity_id
        for merged_id in resolution.merged_entity_ids:
            # 1. Update edges to point to the unified entity
            edges = await session.execute(
                select(Edge).where(
                    (Edge.source_id == merged_id) | (Edge.target_id == merged_id)
                )
            )

            for edge in edges.scalars().all():
                if edge.source_id == merged_id:
                    edge.source_id = target_id
                else:
                    edge.target_id = target_id

            # 2. Update Chunk-Entity links to point to the unified entity
            chunk_links_result = await session.execute(
                select(ChunkEntity).where(ChunkEntity.entity_id == merged_id)
            )
            chunk_links = chunk_links_result.scalars().all()

            for link in chunk_links:
                # Check if target entity already has a link to this chunk
                existing_link_result = await session.execute(
                    select(ChunkEntity).where(
                        ChunkEntity.chunk_id == link.chunk_id,
                        ChunkEntity.entity_id == target_id,
                    )
                )
                if existing_link_result.scalar_one_or_none():
                    await session.delete(link)
                else:
                    link.entity_id = target_id

            # 3. Delete the redundant entity
            merged_entity = await session.get(Entity, merged_id)
            if merged_entity:
                await session.delete(merged_entity)

    async def refine_entity_types(
        self,
        entities: List[Entity],
        domain: str,
    ) -> List[Entity]:
        """Refine entity types using LLM.

        Args:
            entities: List of entities to refine
            domain: Document domain

        Returns:
            Refined entities
        """
        if not self._entity_typer:
            return entities

        typed_entities = []
        for entity in entities:
            typed_entity = TypedEntity(
                entity_id=entity.id,
                name=entity.name,
                description=entity.description or "",
                source_text=None,
                properties=entity.properties or {},
            )
            typed_entities.append(typed_entity)

        refined_entities = await self._entity_typer.type_entities(typed_entities)

        entity_type_map = {str(e.entity_id): e for e in refined_entities}

        for entity in entities:
            entity_id_str = str(entity.id)
            if entity_id_str in entity_type_map:
                refined = entity_type_map[entity_id_str]
                entity.entity_type = refined.entity_type.value
                entity.domain = str(refined.domain) if refined.domain else entity.domain
                entity.confidence = refined.confidence_score

        return entities

    async def cluster_entities(
        self,
        document: Document,
        entities: List[Entity],
        edges: List[Edge],
        session,
    ) -> None:
        """Cluster entities into communities.

        Args:
            document: The document
            entities: List of entities to cluster
            edges: List of edges
            session: Database session
        """
        if not self._clustering_service:
            return

        hierarchy = None

        if entities and edges:
            hierarchy = await self._clustering_service.build_hierarchy(
                list(entities), list(edges)
            )

            # Persist all communities from the hierarchy
            for community_id, community in hierarchy.communities.items():
                existing = (
                    await session.execute(
                        select(community.__class__).where(
                            community.__class__.id == community_id
                        )
                    )
                ).scalar_one_or_none()
                if not existing:
                    session.add(community)

            await session.commit()

            # Update entity community assignments
            for entity_id, community_id in hierarchy.hierarchy.items():
                if community_id:
                    entity = (
                        await session.execute(
                            select(Entity).where(Entity.id == entity_id)
                        )
                    ).scalar_one_or_none()
                    if entity:
                        entity.community_id = community_id

            await session.commit()

        self._logger.info(
            f"Clustered {len(entities)} entities into {len(hierarchy.community_ids) if hierarchy else 0} communities"
        )

    async def extract(
        self,
        document: Document,
        chunks: List[Chunk],
        domain: str,
        use_multi_agent: bool = True,
        recommendation=None,
    ) -> Tuple[List[Entity], List[Edge]]:
        """Extract entities and relationships from a document.

        Args:
            document: The document to extract from
            chunks: List of chunks
            domain: Document domain
            use_multi_agent: If True, try multi-agent first, fallback to gleaning
            recommendation: Adaptive ingestion recommendation (optional)

        Returns:
            Tuple of (entities, edges)
        """
        entities, edges, quality_score, verifications = await self.extract_entities(
            document=document,
            chunks=chunks,
            domain=domain,
            use_multi_agent=use_multi_agent,
            recommendation=recommendation,
        )

        # Persist entities and edges to database
        if entities:
            await self._persist_entities_and_edges(
                document=document,
                chunks=chunks,
                entities=entities,
                edges=edges,
            )

        return entities, edges

    async def _persist_entities_and_edges(
        self,
        document: Document,
        chunks: List[Chunk],
        entities: List[Entity],
        edges: List[Edge],
    ) -> None:
        """Persist extracted entities and edges to database.

        Args:
            document: The document
            chunks: List of chunks
            entities: Entities to persist
            edges: Edges to persist
        """
        async with self._vector_store.get_session() as session:
            # Create chunk to entity mapping (assume first chunk for now)
            primary_chunk = chunks[0] if chunks else None

            # Persist entities and create chunk-entity links
            for entity in entities:
                session.add(entity)
                if primary_chunk:
                    chunk_entity = ChunkEntity(
                        chunk_id=primary_chunk.id,
                        entity_id=entity.id,
                    )
                    session.add(chunk_entity)

            # Persist edges
            for edge in edges:
                session.add(edge)

            await session.commit()

            self._logger.info(
                f"Persisted {len(entities)} entities and {len(edges)} edges to database"
            )

    async def resolve_and_cluster(
        self,
        document: Document,
        entities: List[Entity],
        quality_assurance_service: QualityAssuranceService | None = None,
    ) -> None:
        """Resolve duplicate entities and cluster them into communities.

        Args:
            document: The document
            entities: List of entities to process
            quality_assurance_service: Optional QA service for routing to review
        """
        async with self._vector_store.get_session() as session:
            resolutions = await self.resolve_entities(
                document=document,
                entities=entities,
                session=session,
            )

            for resolution in resolutions:
                if resolution.human_review_required:
                    chunk_text = ""
                    chunk_result = await session.execute(
                        select(Chunk)
                        .join(ChunkEntity, ChunkEntity.chunk_id == Chunk.id)
                        .where(ChunkEntity.entity_id == resolution.entity_id)
                        .limit(1)
                    )
                    chunk = chunk_result.scalar_one_or_none()
                    if chunk:
                        chunk_text = (
                            str(chunk.text)
                            if hasattr(chunk.text, "value")
                            else chunk.text
                        )

                    priority = (
                        3
                        if resolution.confidence_score >= 0.9
                        else (
                            5
                            if resolution.confidence_score >= 0.7
                            else (7 if resolution.confidence_score >= 0.5 else 9)
                        )
                    )
                    if quality_assurance_service:
                        await quality_assurance_service.route_to_review(
                            [
                                {
                                    "item_type": "entity_resolution",
                                    "item_id": resolution.entity_id,
                                    "reason": resolution.grounding_quote,
                                    "priority": priority,
                                    "metadata": {
                                        "document_id": str(document.id),
                                        "merged_entity_ids": [
                                            str(eid)
                                            for eid in resolution.merged_entity_ids
                                        ],
                                        "confidence_score": resolution.confidence_score,
                                        "source_text": chunk_text,
                                    },
                                }
                            ],
                            session,
                        )
                else:
                    await self.merge_entities(resolution, session)

            doc_in_session = await session.get(Document, document.id)
            if doc_in_session:
                doc_in_session.status = "resolved"
                await session.commit()

        await self._cluster_and_update(document, entities)

    async def _cluster_and_update(
        self,
        document: Document,
        entities: List[Entity],
    ) -> None:
        """Cluster entities and update document status."""
        async with self._vector_store.get_session() as session:
            edge_result = await session.execute(
                select(Edge)
                .join(Entity, Edge.source_id == Entity.id)
                .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
                .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
                .where(Chunk.document_id == document.id)
            )
            edges = edge_result.scalars().all()

        await self.cluster_entities(
            document=document,
            entities=entities,
            edges=list(edges),
            session=session,
        )

        # Refresh entities from database to get updated community_ids
        async with self._vector_store.get_session() as session:
            from sqlalchemy import select

            refreshed_result = await session.execute(
                select(Entity)
                .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
                .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
                .where(Chunk.document_id == document.id)
            )
            refreshed_entities = refreshed_result.scalars().all()

        # Generate community summaries after clustering with refreshed entities
        await self._generate_community_summaries(
            document=document,
            entities=list(refreshed_entities),
            edges=list(edges),
            session=session,
        )

    async def _generate_community_summaries(
        self,
        document: Document,
        entities: List[Entity],
        edges: List[Edge],
        session,
    ) -> None:
        """Generate and store hierarchical community summaries.

        Args:
            document: The document
            entities: List of entities
            edges: List of edges
            session: Database session
        """
        if not entities or not edges:
            self._logger.info("No entities or edges to summarize")
            return

        try:
            # Initialize summarizer if needed
            if not self._community_summarizer:
                self._community_summarizer = CommunitySummarizer()

            # Prepare entity data for summarizer
            entity_data = []
            for entity in entities:
                entity_data.append(
                    {
                        "name": entity.name,
                        "type": entity.entity_type,
                        "community_id": str(entity.community_id)
                        if entity.community_id
                        else None,
                        "id": str(entity.id),
                    }
                )

            # Prepare relationship data
            relationship_data = []
            for edge in edges:
                # Get source and target entity names
                source_entity = next(
                    (e for e in entities if e.id == edge.source_id), None
                )
                target_entity = next(
                    (e for e in entities if e.id == edge.target_id), None
                )

                if source_entity and target_entity:
                    relationship_data.append(
                        {
                            "source": source_entity.name,
                            "target": target_entity.name,
                            "type": edge.edge_type,
                            "description": f"{source_entity.name} {edge.edge_type} {target_entity.name}",
                        }
                    )

            self._logger.info(
                f"Generating community summaries for {len(entity_data)} entities "
                f"and {len(relationship_data)} relationships"
            )

            # Generate multi-level summary
            summary = await self._community_summarizer.generate_multi_level_summary(
                document_id=str(document.id),
                entities=entity_data,
                relationships=relationship_data,
            )

            # Store summary in document metadata
            summary_dict = {
                "macro_communities": [
                    {
                        "id": c.community_id,
                        "name": c.name,
                        "level": c.level.value,
                        "summary": c.summary,
                        "key_entities": c.key_entities,
                        "key_relationships": c.key_relationships,
                        "entity_count": c.entity_count,
                        "coherence_score": c.coherence_score,
                    }
                    for c in summary.macro_communities
                ],
                "meso_communities": [
                    {
                        "id": c.community_id,
                        "name": c.name,
                        "level": c.level.value,
                        "summary": c.summary,
                        "key_entities": c.key_entities,
                        "entity_count": c.entity_count,
                    }
                    for c in summary.meso_communities
                ],
                "micro_communities": [
                    {
                        "id": c.community_id,
                        "name": c.name,
                        "level": c.level.value,
                        "summary": c.summary,
                        "key_entities": c.key_entities,
                        "entity_count": c.entity_count,
                    }
                    for c in summary.micro_communities
                ],
                "hierarchy_tree": summary.hierarchy_tree,
                "total_communities": (
                    len(summary.macro_communities)
                    + len(summary.meso_communities)
                    + len(summary.micro_communities)
                ),
            }

            # Update document with community summary
            if not document.metadata:
                document.metadata = {}
            document.metadata["community_summary"] = summary_dict
            document.metadata["community_summary_generated_at"] = (
                datetime.utcnow().isoformat()
            )

            await session.commit()

            self._logger.info(
                f"✅ Community summaries generated: "
                f"{len(summary.macro_communities)} macro, "
                f"{len(summary.meso_communities)} meso, "
                f"{len(summary.micro_communities)} micro communities"
            )

        except Exception as e:
            self._logger.error(
                f"Error generating community summaries: {e}", exc_info=True
            )
            # Don't fail the entire pipeline if summarization fails
