"""Main ReAct loop orchestrator."""

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from urllib.parse import quote
from uuid import UUID, uuid4

import numpy as np

from knowledge_base.common.gateway import GatewayClient, EnhancedGateway
from knowledge_base.common.resilient_gateway import ResilientGatewayClient
from knowledge_base.config.constants import (
    DOMAIN_CONFIDENCE_THRESHOLD,
    ENTITY_SIMILARITY_THRESHOLD,
    HALLUCINATION_THRESHOLD,
    MIN_EXTRACTION_QUALITY_SCORE,
    MIN_COMMUNITY_SIZE,
    ROTATION_DELAY,
)
from knowledge_base.domain.detection import DomainDetector
from knowledge_base.domain.domain_models import DomainConfig
from knowledge_base.intelligence.v1.multi_agent_extractor import (
    ExtractionQualityScore,
)
from knowledge_base.intelligence.v1.adaptive_ingestion_engine import (
    AdaptiveIngestionEngine,
    PipelineRecommendation,
)
from knowledge_base.intelligence.v1.hallucination_detector import (
    HallucinationDetector,
    EntityVerification,
    RiskLevel,
)
from knowledge_base.intelligence.v1.cross_domain_detector import (
    CrossDomainDetector,
)
from knowledge_base.ingestion.v1.embedding_client import EmbeddingClient
from knowledge_base.ingestion.v1.gleaning_service import (
    ExtractionResult,
    GleaningService,
)
from knowledge_base.intelligence.v1.clustering_service import (
    ClusteringService,
)
from knowledge_base.intelligence.v1.resolution_agent import (
    EntityResolution,
    ResolutionAgent,
)
from knowledge_base.intelligence.v1.synthesis_agent import SynthesisAgent
from knowledge_base.intelligence.v1.entity_typing_service import (
    EntityTyper,
    EntityTypingConfig,
)
from knowledge_base.observability import Observability
from knowledge_base.intelligence.v1.domain_schema_service import SchemaRegistry
from knowledge_base.orchestration.domain_detection_service import DomainDetectionService
from knowledge_base.orchestration.document_pipeline_service import (
    DocumentPipelineService,
)
from knowledge_base.orchestration.entity_pipeline_service import (
    EntityPipelineService,
)
from knowledge_base.persistence.v1.schema import (
    Chunk,
    Community,
    Document,
    Edge,
    EdgeType,
    Entity,
    ChunkEntity,
    ReviewQueue,
    ReviewStatus,
)
from knowledge_base.persistence.v1.vector_store import VectorStore
from knowledge_base.persistence.v1.graph_store import GraphStore
from sqlalchemy import select

logger = logging.getLogger(__name__)


class IngestionOrchestrator:
    """ReAct loop orchestrator for knowledge ingestion."""

    def __init__(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        log_broadcast: Optional[Callable[[str], Any]] = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            progress_callback: Optional callback for progress updates.
            log_broadcast: Optional callback for broadcasting log messages via WebSocket.
        """
        self._progress_callback = progress_callback
        self._log_broadcast = log_broadcast
        # Set up the global WebSocket broadcast function for extraction logging
        if log_broadcast:
            from knowledge_base.intelligence.v1.extraction_logging import (
                set_websocket_broadcast,
            )

            set_websocket_broadcast(log_broadcast)
        self._observability: Observability | None = None
        self._gateway: GatewayClient | None = None
        self._vector_store: VectorStore | None = None
        self._embedding_client: EmbeddingClient | None = None
        self._gleaning_service: GleaningService | None = None
        self._clustering_service: ClusteringService | None = None
        self._resolution_agent: ResolutionAgent | None = None
        self._synthesis_agent: SynthesisAgent | None = None
        self._hallucination_detector: HallucinationDetector | None = None
        self._cross_domain_detector: CrossDomainDetector | None = None
        self._entity_typer: EntityTyper | None = None
        self._schema_registry: SchemaRegistry | None = None
        self._domain_detector: DomainDetector | None = None
        self._domain_service: DomainDetectionService | None = None
        self._adaptive_engine: AdaptiveIngestionEngine | None = None
        self._document_service: DocumentPipelineService | None = None
        self._entity_pipeline_service: EntityPipelineService | None = None

    async def initialize(self) -> None:
        """Initialize all components."""
        self._observability = Observability()
        # Initialize gateway with continuous rotation enabled for robust model failover
        from knowledge_base.common.resilient_gateway import ResilientGatewayConfig

        gateway_config = ResilientGatewayConfig(
            continuous_rotation_enabled=True,  # Enable continuous model rotation
            rotation_delay=ROTATION_DELAY,
        )
        self._gateway = ResilientGatewayClient(config=gateway_config)
        self._embedding_client = EmbeddingClient()
        self._vector_store = VectorStore()
        # GraphStore is created lazily in methods that have a session
        self._graph_store = None
        self._gleaning_service = GleaningService(self._gateway)
        self._clustering_service = ClusteringService()
        self._resolution_agent = ResolutionAgent(
            gateway=self._gateway, vector_store=self._vector_store
        )
        self._synthesis_agent = SynthesisAgent(gateway=self._gateway)
        self._hallucination_detector = HallucinationDetector()
        self._cross_domain_detector = CrossDomainDetector()
        self._entity_typer = EntityTyper(
            gateway=self._gateway,
            config=EntityTypingConfig(
                confidence_threshold=0.6,
                max_few_shot_examples=5,
                temperature=0.2,
                enable_domain_awareness=True,
            ),
        )

        await self._vector_store.initialize()

        await self._vector_store.initialize()
        self._domain_detector = DomainDetector(
            llm_client=self._gateway,
            config=DomainConfig(
                min_confidence=DOMAIN_CONFIDENCE_THRESHOLD,
                max_predictions=3,
                enable_keyword_screening=True,
                enable_llm_analysis=True,
                keyword_threshold=0.3,
            ),
        )
        self._domain_service = DomainDetectionService()
        await self._domain_service.initialize()

        # Initialize adaptive ingestion engine for intelligent pipeline optimization
        self._adaptive_engine = AdaptiveIngestionEngine(gateway=self._gateway)

        # Initialize document pipeline service
        self._document_service = DocumentPipelineService()
        await self._document_service.initialize()

        # Initialize entity pipeline service
        self._entity_pipeline_service = EntityPipelineService()
        await self._entity_pipeline_service.initialize(
            vector_store=self._vector_store,
            gateway=self._gateway,
        )
        self._entity_pipeline_service.set_extractors(
            multi_agent_extractor=None,
            entity_resolver=self._resolution_agent,
            entity_typer=self._entity_typer,
            clustering_service=self._clustering_service,
            gleaning_service=self._gleaning_service,
            hallucination_detector=self._hallucination_detector,
        )

        logger.info("IngestionOrchestrator initialized successfully")

    async def _send_progress(self, progress_data: dict[str, Any]) -> None:
        """Send progress update.

        Args:
            progress_data: Progress information to send.
        """
        if self._progress_callback:
            if asyncio.iscoroutinefunction(self._progress_callback):
                await self._progress_callback(progress_data)
            else:
                self._progress_callback(progress_data)

    async def _partition_document(
        self,
        document: Document,
        file_path: str | Path,
    ) -> Document:
        """Partition document into chunks.

        Args:
            document: Document record.
            file_path: Path to file.

        Returns:
            Updated document.
        """
        obs = self._observability

        async with obs.trace_context(
            "partition_document", document_id=str(document.id)
        ):
            # Use DocumentPipelineService to partition the document
            from knowledge_base.partitioning.semantic_chunker import SemanticChunker

            # Read the file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Use the chunker from DocumentPipelineService
            chunker = self._document_service._chunker
            chunks = chunker.partition(content)

            async with self._vector_store.get_session() as session:
                for chunk_data in chunks:
                    chunk = Chunk(
                        id=uuid4(),
                        document_id=document.id,
                        text=chunk_data.text,
                        chunk_index=chunk_data.chunk_index,
                        page_number=chunk_data.page_number,
                        token_count=chunk_data.token_count,
                        chunk_metadata=chunk_data.metadata,
                    )
                    session.add(chunk)

                document = await session.merge(document)
                document.status = "partitioned"
                await session.commit()
                # Refresh document from session
                refreshed_doc = await session.get(Document, document.id)
                if refreshed_doc:
                    document = refreshed_doc

            obs.log_metric("chunks_created", len(chunks), document_id=str(document.id))

            return document

    async def _extract_knowledge(
        self,
        document: Document,
        use_multi_agent: bool = True,
        recommendation: PipelineRecommendation | None = None,
    ) -> Document:
        """Extract entities, edges, and temporal claims.

        Args:
            document: Document record.
            use_multi_agent: If True, try multi-agent first, fallback to gleaning.
            recommendation: Adaptive ingestion recommendation (optional).

        Returns:
            Updated document.
        """
        obs = self._observability

        async with obs.trace_context("extract_knowledge", document_id=str(document.id)):
            async with self._vector_store.get_session() as session:
                chunk_result = await session.execute(
                    select(Chunk).where(Chunk.document_id == document.id)
                )
                chunks = chunk_result.scalars().all()

                domain = await self._domain_service.detect_domain(document)

                # Use entity pipeline service for extraction
                (
                    entities,
                    edges,
                    quality_score,
                    verifications,
                ) = await self._entity_pipeline_service.extract_entities(
                    document=document,
                    chunks=list(chunks),
                    domain=domain,
                    use_multi_agent=use_multi_agent,
                    recommendation=recommendation,
                )

                # Deduplicate entities by URI
                uri_to_id: dict[str, UUID] = {}
                unique_entities = []

                for entity in entities:
                    if entity.uri in uri_to_id:
                        # Already saw this entity in this extraction
                        continue

                    # Query for existing entity with same URI in database
                    result = await session.execute(
                        select(Entity).where(Entity.uri == entity.uri)
                    )
                    existing = result.scalar_one_or_none()

                    if existing:
                        uri_to_id[entity.uri] = existing.id
                        # Update extraction entities to use the existing ID for edge mapping
                        entity.id = existing.id
                    else:
                        session.add(entity)
                        uri_to_id[entity.uri] = entity.id
                        unique_entities.append(entity)

                await session.flush()

                # Map edges using potentially updated IDs from deduplication
                for edge in edges:
                    if (
                        edge.source_id
                        and edge.target_id
                        and edge.source_id != edge.target_id
                    ):
                        session.add(edge)

                # Map entities to chunks (for gleaning extraction)
                if not quality_score:  # Gleaning mode
                    known_uri_to_id: dict[str, UUID] = {}
                    for chunk in chunks:
                        chunk_unique_entity_ids: set[UUID] = set()
                        for entity in entities:
                            if entity.uri in known_uri_to_id:
                                target_id = known_uri_to_id[entity.uri]
                                chunk_unique_entity_ids.add(target_id)
                                continue

                            if entity.uri in uri_to_id:
                                chunk_unique_entity_ids.add(uri_to_id[entity.uri])
                                known_uri_to_id[entity.uri] = uri_to_id[entity.uri]

                        # Map entities to this chunk
                        for entity_id in chunk_unique_entity_ids:
                            chunk_entity = ChunkEntity(
                                chunk_id=chunk.id, entity_id=entity_id
                            )
                            session.add(chunk_entity)

                if quality_score:
                    obs.log_metric(
                        "multi_agent_extraction_quality",
                        quality_score.overall_score,
                        document_id=str(document.id),
                    )

                obs.log_metric(
                    "entities_extracted",
                    len(entities),
                    document_id=str(document.id),
                )
                obs.log_metric(
                    "edges_extracted",
                    len(edges),
                    document_id=str(document.id),
                )

                # Route hallucinated entities to review
                if verifications:
                    for verification in verifications:
                        if verification.is_hallucinated or verification.risk_level in (
                            RiskLevel.HIGH,
                            RiskLevel.CRITICAL,
                        ):
                            await self._route_to_review([verification], document, "")

                # Detect cross-domain relationships
                if self._cross_domain_detector:
                    entity_result = await session.execute(
                        select(Entity)
                        .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
                        .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
                        .where(Chunk.document_id == document.id)
                    )
                    entities = entity_result.scalars().all()

                    edge_result = await session.execute(
                        select(Edge)
                        .join(Entity, Edge.source_id == Entity.id)
                        .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
                        .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
                        .where(Chunk.document_id == document.id)
                    )
                    existing_edges = edge_result.scalars().all()

                    if entities:
                        entities_data = [
                            {
                                "id": str(e.id),
                                "name": e.name,
                                "entity_type": e.entity_type,
                                "domain": getattr(e, "domain", "general") or "general",
                                "properties": e.properties or {},
                            }
                            for e in entities
                        ]
                        edges_data = [
                            {
                                "id": str(e.id),
                                "source_id": str(e.source_id),
                                "target_id": str(e.target_id),
                                "edge_type": e.edge_type,
                            }
                            for e in existing_edges
                        ]

                        document_domain = await self._domain_service.detect_domain(
                            document
                        )
                        cross_domain_edges = await self._cross_domain_detector.detect_cross_domain_relationships(
                            entities=entities_data,
                            edges=edges_data,
                            document_domain=document_domain,
                        )

                        for edge_dict in cross_domain_edges:
                            new_edge = Edge(
                                id=uuid4(),
                                source_id=UUID(edge_dict["source_id"]),
                                target_id=UUID(edge_dict["target_id"]),
                                edge_type=edge_dict["edge_type"],
                                properties=edge_dict.get("properties", {}),
                                confidence=edge_dict.get("confidence", 0.5),
                                provenance=edge_dict.get(
                                    "provenance", "cross_domain_detection"
                                ),
                                domain=edge_dict.get("domain", document_domain),
                            )
                            session.add(new_edge)

                        if cross_domain_edges:
                            await session.commit()
                            obs.log_metric(
                                "cross_domain_edges_added",
                                len(cross_domain_edges),
                                document_id=str(document.id),
                            )

                doc_in_session = await session.get(Document, document.id)
                if doc_in_session:
                    doc_in_session.status = "extracted"
                    await session.commit()
                    refreshed_doc = await session.get(Document, document.id)
                    if refreshed_doc:
                        document = refreshed_doc

                obs.log_metric(
                    "chunks_processed", len(chunks), document_id=str(document.id)
                )

                return document

    async def _resolve_entities(self, document: Document) -> Document:
        """Resolve duplicate entities.

        Args:
            document: Document record.

        Returns:
            Updated document.
        """
        obs = self._observability

        async with obs.trace_context("resolve_entities", document_id=str(document.id)):
            async with self._vector_store.get_session() as session:
                result = await session.execute(
                    select(Entity)
                    .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
                    .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
                    .where(Chunk.document_id == document.id)
                )
                entities = result.scalars().all()

                # Use entity pipeline service for resolution
                resolutions = await self._entity_pipeline_service.resolve_entities(
                    document=document,
                    entities=list(entities),
                    session=session,
                )

                # Process resolutions
                for resolution in resolutions:
                    if resolution.human_review_required:
                        # Get chunk context for review
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

                        await self._add_to_review_queue(
                            document.id,
                            resolution.entity_id,
                            resolution.merged_entity_ids,
                            resolution.confidence_score,
                            resolution.grounding_quote,
                            chunk_text,
                        )
                    else:
                        await self._entity_pipeline_service.merge_entities(
                            resolution, session
                        )

            return document

    async def _refine_entity_types(self, document: Document) -> Document:
        obs = self._observability

        async with obs.trace_context(
            "refine_entity_types", document_id=str(document.id)
        ):
            async with self._vector_store.get_session() as session:
                entity_result = await session.execute(
                    select(Entity)
                    .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
                    .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
                    .where(Chunk.document_id == document.id)
                )
                entities = entity_result.scalars().all()

            if not entities:
                return document

            # Use entity pipeline service for type refinement
            domain = await self._domain_service.detect_domain(document)
            refined_entities = await self._entity_pipeline_service.refine_entity_types(
                entities=list(entities),
                domain=domain,
            )

            async with self._vector_store.get_session() as session:
                for entity in refined_entities:
                    # Merge the updated entity
                    await session.merge(entity)

                await session.commit()

                doc_in_session = await session.get(Document, document.id)
                if doc_in_session:
                    doc_in_session.status = "typed"
                    await session.commit()
                    refreshed_doc = await session.get(Document, document.id)
                    if refreshed_doc:
                        document = refreshed_doc

            obs.log_metric(
                "entities_refined",
                len(refined_entities),
                document_id=str(document.id),
            )

            return document

    async def _validate_entities_against_schema(self, document: Document) -> Document:
        obs = self._observability

        async with obs.trace_context(
            "validate_entities_against_schema", document_id=str(document.id)
        ):
            async with self._vector_store.get_session() as session:
                registry = SchemaRegistry(session)
                schema = await registry.get_by_name(document.domain)

                if not schema:
                    logger.info(f"No schema found for domain: {document.domain}")
                    return document

                entity_result = await session.execute(
                    select(Entity)
                    .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
                    .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
                    .where(Chunk.document_id == document.id)
                )
                entities = entity_result.scalars().all()

            if not entities:
                return document

            validated_count = 0
            missing_attributes_added = 0

            for entity in entities:
                if not entity.entity_type:
                    continue

                try:
                    async with self._vector_store.get_session() as session:
                        registry = SchemaRegistry(session)
                        entity_type_def = await registry.apply_inheritance(
                            document.domain, entity.entity_type
                        )

                        if not entity_type_def:
                            continue

                        entity_properties = entity.properties or {}
                        properties_modified = False

                        for required_attr in entity_type_def.required_attributes:
                            if required_attr not in entity_properties:
                                attr_def = entity_type_def.attributes.get(required_attr)
                                if attr_def and attr_def.default_value is not None:
                                    entity_properties[required_attr] = (
                                        attr_def.default_value
                                    )
                                    missing_attributes_added += 1
                                    properties_modified = True

                        if properties_modified:
                            entity_to_update = await session.get(Entity, entity.id)
                            if entity_to_update:
                                entity_to_update.properties = entity_properties
                                await session.commit()

                        validated_count += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to validate entity {entity.id} against schema: {e}"
                    )
                    continue

            obs.log_metric(
                "entities_validated",
                validated_count,
                document_id=str(document.id),
            )
            obs.log_metric(
                "missing_attributes_added",
                missing_attributes_added,
                document_id=str(document.id),
            )

            return document

    async def _cluster_entities(self, document: Document) -> Document:
        """Cluster entities into communities.

        Args:
            document: Document record.

        Returns:
            Updated document.
        """
        obs = self._observability

        async with obs.trace_context("cluster_entities", document_id=str(document.id)):
            async with self._vector_store.get_session() as session:
                entity_result = await session.execute(
                    select(Entity)
                    .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
                    .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
                    .where(Chunk.document_id == document.id)
                )
                entities = entity_result.scalars().all()

                edge_result = await session.execute(
                    select(Edge)
                    .join(Entity, Edge.source_id == Entity.id)
                    .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
                    .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
                    .where(Chunk.document_id == document.id)
                )
                edges = edge_result.scalars().all()

            # Use entity pipeline service for clustering
            await self._entity_pipeline_service.cluster_entities(
                document=document,
                entities=list(entities),
                edges=list(edges),
                session=session,
            )

            obs.log_metric(
                "entities_clustered",
                len(entities),
                document_id=str(document.id),
            )

            return document

    async def _generate_reports(self, document: Document) -> Document:
        """Generate intelligence reports.

        Args:
            document: Document record.

        Returns:
            Updated document.
        """
        obs = self._observability

        async with obs.trace_context("generate_reports", document_id=str(document.id)):
            async with self._vector_store.get_session() as session:
                community_result = await session.execute(select(Community))
                communities = community_result.scalars().all()

                entity_result = await session.execute(select(Entity))
                entities = entity_result.scalars().all()

                edge_result = await session.execute(select(Edge))
                edges = edge_result.scalars().all()

            if communities:
                hierarchy = {c.id: c for c in communities}

                reports = await self._synthesis_agent.generate_intelligence_report(
                    hierarchy,
                    list(entities),
                    list(edges),
                )

                async with self._vector_store.get_session() as session:
                    for community in communities:
                        if community.id in reports:
                            report = reports[community.id]
                            community.summary = report.summary
                            await session.commit()

            return document

    async def global_deduplicate_entities(self) -> dict[str, Any]:
        """Perform a global entity resolution sweep across the entire knowledge base."""
        async with self._vector_store.get_session() as session:
            # 1. Fetch all entities
            result = await session.execute(select(Entity))
            all_entities = result.scalars().all()
            total = len(all_entities)
            merged_count = 0
            review_count = 0

            logger.info(f"Starting global deduplication for {total} entities")

            for i, entity in enumerate(all_entities):
                if i % 10 == 0:
                    logger.info(
                        f"Deduplication progress: {i}/{total} candidates processed..."
                    )

                # Similar logic to _resolve_entities but targeting the whole DB
                embedding = entity.embedding
                if embedding is None:
                    continue

                # Ensure entity still exists (might have been merged away in this loop)
                current_entity = await session.get(Entity, entity.id)
                if not current_entity:
                    continue

                similar = await self._vector_store.search_similar_entities(
                    embedding.tolist() if hasattr(embedding, "tolist") else embedding,
                    limit=5,
                    similarity_threshold=ENTITY_SIMILARITY_THRESHOLD,
                )

                candidate_ids = []
                for s in similar:
                    # Force string conversion before calling UUID constructor
                    # to avoid issues with raw asyncpg UUID objects
                    s_id_str = str(s["id"])
                    if s_id_str != str(entity.id):
                        try:
                            candidate_ids.append(UUID(s_id_str))
                        except (ValueError, TypeError):
                            continue

                if not candidate_ids:
                    continue

                candidate_result = await session.execute(
                    select(Entity).where(Entity.id.in_(candidate_ids))
                )
                candidates = candidate_result.scalars().all()

                if candidates:
                    # Get any chunk context for grounding
                    chunk_result = await session.execute(
                        select(Chunk)
                        .join(ChunkEntity, ChunkEntity.chunk_id == Chunk.id)
                        .where(ChunkEntity.entity_id == entity.id)
                        .limit(1)
                    )
                    chunk = chunk_result.scalar_one_or_none()
                    chunk_text = (
                        chunk.text
                        if chunk
                        else "Global cleanup sweep - no specific chunk context available for this entity."
                    )

                    resolution = await self._resolution_agent.resolve_entity(
                        entity, candidates, chunk_text
                    )

                    if resolution.merged_entity_ids:
                        if resolution.human_review_required:
                            await self._add_to_review_queue(
                                None,
                                entity.id,
                                resolution.merged_entity_ids,
                                resolution.confidence_score,
                                resolution.grounding_quote,
                                chunk_text,
                            )
                            review_count += 1
                        else:
                            await self._merge_entities(resolution)
                            merged_count += len(resolution.merged_entity_ids)

            return {
                "total_processed": total,
                "merged_entities": merged_count,
                "pending_review": review_count,
            }

    async def _add_to_review_queue(
        self,
        document_id: UUID,
        entity_id: UUID,
        merged_entity_ids: list[UUID],
        confidence_score: float,
        grounding_quote: str,
        source_text: str,
    ) -> None:
        """Add entity resolution to review queue.

        Args:
            document_id: Source document ID.
            entity_id: Target entity ID.
            merged_entity_ids: IDs of entities to be merged.
            confidence_score: Confidence in the resolution decision.
            grounding_quote: Quote supporting the resolution.
            source_text: Original source text.
        """
        if not merged_entity_ids:
            return

        if confidence_score >= 0.9:
            priority = 3
        elif confidence_score >= 0.7:
            priority = 5
        elif confidence_score >= 0.5:
            priority = 7
        else:
            priority = 9

        review_item = ReviewQueue(
            id=uuid4(),
            item_type="entity_resolution",
            entity_id=entity_id,
            document_id=document_id,
            merged_entity_ids=merged_entity_ids,
            confidence_score=confidence_score,
            grounding_quote=grounding_quote,
            source_text=source_text,
            status=ReviewStatus.PENDING,
            priority=priority,
        )

        async with self._vector_store.get_session() as session:
            session.add(review_item)
            await session.commit()

        self._observability.log_metric(
            "review_queue_items_added",
            1,
            document_id=str(document_id),
            priority=priority,
        )

    async def _update_document(self, document: Document) -> None:
        """Update document in database.

        Args:
            document: Document to update.
        """
        async with self._vector_store.get_session() as session:
            await session.merge(document)
            await session.commit()

    def _get_mime_type(self, path: Path) -> str:
        """Get MIME type for file.

        Args:
            path: File path.

        Returns:
            MIME type string.
        """
        import mimetypes

        mime_type, _ = mimetypes.guess_type(str(path))
        return mime_type or "application/octet-stream"

    async def _emit_progress(
        self, stage: int, status: str, message: str, **kwargs
    ) -> None:
        """Emit progress update.

        Args:
            stage: Pipeline stage number (1-9).
            status: Status of the stage (started, completed, failed).
            message: Human-readable message.
            **kwargs: Additional context data.
        """
        progress_data = {
            "stage": stage,
            "status": status,
            "message": message,
            "total_stages": 9,
            **kwargs,
        }
        await self._send_progress(progress_data)

    async def _route_to_review(
        self,
        verifications: List[EntityVerification],
        document: Document,
        source_text: str,
    ) -> None:
        """Route low-quality entities to review queue."""
        for verification in verifications:
            if verification.is_hallucinated or verification.risk_level in (
                RiskLevel.HIGH,
                RiskLevel.CRITICAL,
            ):
                priority = 9 if verification.risk_level == RiskLevel.CRITICAL else 7

                review_item = ReviewQueue(
                    id=uuid4(),
                    item_type="entity_verification",
                    document_id=document.id,
                    confidence_score=verification.overall_confidence,
                    grounding_quote=", ".join(verification.hallucination_reasons),
                    source_text=source_text,
                    status=ReviewStatus.PENDING,
                    priority=priority,
                    metadata={
                        "entity_name": verification.entity_name,
                        "entity_type": verification.entity_type,
                        "risk_level": verification.risk_level.value,
                        "unsupported_count": verification.unsupported_count,
                        "total_attributes": verification.total_attributes,
                    },
                )

                async with self._vector_store.get_session() as session:
                    session.add(review_item)
                    await session.commit()

                self._observability.log_metric(
                    "review_queue_items_added",
                    1,
                    document_id=str(document.id),
                    priority=priority,
                )

    async def process_document(
        self,
        file_path: str | Path,
        document_name: str | None = None,
        domain: str | None = None,
    ) -> Document:
        """Process a document through the full ingestion pipeline.

        Args:
            file_path: Path to document file.
            document_name: Optional document name.
            domain: Optional domain for the document.

        Returns:
            Processed document.
        """
        obs = self._observability

        try:
            async with obs.trace_context(
                "document_ingestion",
                file_path=str(file_path),
            ):
                await self._emit_progress(1, "started", "Creating document record")
                document = await self._create_document(file_path, document_name)
                await self._emit_progress(1, "completed", "Document record created")

                await self._emit_progress(
                    2, "started", "Partitioning document into chunks"
                )
                document = await self._partition_document(document, file_path)
                await self._emit_progress(
                    2, "completed", "Document partitioned into chunks"
                )

                # Adaptive analysis: Let LLM decide optimal processing strategy
                await self._emit_progress(
                    2.5, "started", "Analyzing document complexity"
                )

                # Combine first few chunks for analysis (avoid token limits)
                async with self._vector_store.get_session() as session:
                    chunk_result = await session.execute(
                        select(Chunk).where(Chunk.document_id == document.id).limit(3)
                    )
                    sample_chunks = chunk_result.scalars().all()

                    sample_text = " ".join([chunk.text for chunk in sample_chunks])

                    # Get adaptive recommendation
                    recommendation = await self._adaptive_engine.analyze_document(
                        document_text=sample_text,
                        document_name=document.name,
                        file_size_bytes=sum(
                            [len(chunk.text) for chunk in sample_chunks]
                        ),
                    )

                await self._emit_progress(
                    2.5,
                    "completed",
                    f"Analysis complete: {recommendation.complexity.value}, "
                    f"{recommendation.expected_entity_count} entities expected, "
                    f"{recommendation.estimated_processing_time} processing time",
                )

                await self._emit_progress(
                    3, "started", "Extracting knowledge (entities and edges)"
                )
                document = await self._extract_knowledge(document, recommendation)
                await self._emit_progress(
                    3, "completed", "Knowledge extraction complete"
                )

                await self._emit_progress(3.5, "started", "Refining entity types")
                document = await self._refine_entity_types(document)
                await self._emit_progress(
                    3.5, "completed", "Entity type refinement complete"
                )

                await self._emit_progress(
                    3.75, "started", "Validating entities against schema"
                )
                document = await self._validate_entities_against_schema(document)
                await self._emit_progress(
                    3.75, "completed", "Entity schema validation complete"
                )

                await self._emit_progress(4, "started", "Embedding chunks and entities")
                document = await self._embed_content(document)
                await self._emit_progress(4, "completed", "Embedding complete")

                await self._emit_progress(5, "started", "Resolving duplicate entities")
                document = await self._resolve_entities(document)
                await self._emit_progress(5, "completed", "Entity resolution complete")

                await self._emit_progress(
                    6, "started", "Clustering entities into communities"
                )
                document = await self._cluster_entities(document)
                await self._emit_progress(6, "completed", "Entity clustering complete")

                await self._emit_progress(
                    7, "started", "Generating intelligence reports"
                )
                document = await self._generate_reports(document)
                await self._emit_progress(
                    7, "completed", "Intelligence reports generated"
                )

                await self._emit_progress(
                    8, "started", "Updating domain for document and entities"
                )
                final_domain = (
                    domain
                    if domain is not None
                    else await self._domain_service.detect_domain(document)
                )
                async with self._vector_store.get_session() as session:
                    doc_to_update = await session.get(Document, document.id)
                    if doc_to_update:
                        doc_to_update.domain = (
                            str(final_domain) if final_domain else None
                        )
                    entity_result = await session.execute(
                        select(Entity)
                        .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
                        .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
                        .where(Chunk.document_id == document.id)
                    )
                    for entity in entity_result.scalars().all():
                        entity.domain = str(final_domain) if final_domain else None
                    edge_result = await session.execute(
                        select(Edge)
                        .join(Entity, Edge.source_id == Entity.id)
                        .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
                        .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
                        .where(Chunk.document_id == document.id)
                    )
                    for edge in edge_result.scalars().all():
                        edge.domain = str(final_domain) if final_domain else None
                    await session.commit()
                await self._emit_progress(8, "completed", "Domain updated successfully")

                await self._emit_progress(9, "started", "Finalizing document status")
                async with self._vector_store.get_session() as session:
                    doc_to_update = await session.get(Document, document.id)
                    if doc_to_update:
                        doc_to_update.status = "completed"
                        await session.commit()
                        refreshed_doc = await session.get(Document, document.id)
                        if refreshed_doc:
                            document = refreshed_doc
                await self._emit_progress(
                    9, "completed", "Document processing complete"
                )

                obs.log_event(
                    "document_processing_completed",
                    document_id=str(document.id),
                    document_name=document.name,
                )

                return document

        except Exception as e:
            doc_id = getattr(document, "id", None) if "document" in dir() else None
            async with self._vector_store.get_session() as session:
                if doc_id:
                    doc_to_update = await session.get(Document, doc_id)
                    if doc_to_update:
                        doc_to_update.status = "failed"
                        await session.commit()

            obs.log_event(
                "document_processing_failed",
                document_id=str(doc_id) if doc_id else "unknown",
                error=str(e),
                level="error",
            )
            raise

    async def _create_document(
        self, file_path: str | Path, document_name: str | None = None
    ) -> Document:
        """Create a document record in the database.

        Args:
            file_path: Path to the document file.
            document_name: Optional document name override.

        Returns:
            Created document.
        """
        obs = self._observability

        async with obs.trace_context("create_document", file_path=str(file_path)):
            path = Path(file_path) if isinstance(file_path, str) else file_path
            name = document_name or path.name
            mime_type = self._get_mime_type(path)

            document = Document(
                id=uuid4(),
                name=name,
                source_uri=str(path),
                mime_type=mime_type,
                status="pending",
                doc_metadata={"source": str(path)},
            )

            async with self._vector_store.get_session() as session:
                session.add(document)
                await session.commit()
                await session.refresh(document)

            obs.log_event(
                "document_created",
                document_id=str(document.id),
                document_name=document.name,
            )

            return document

    async def _embed_content(self, document: Document) -> Document:
        """Embed chunks and entities.

        Args:
            document: Document record.

        Returns:
            Updated document.
        """
        obs = self._observability

        async with obs.trace_context("embed_content", document_id=str(document.id)):
            async with self._vector_store.get_session() as session:
                chunk_result = await session.execute(
                    select(Chunk).where(Chunk.document_id == document.id)
                )
                chunks = chunk_result.scalars().all()

                entity_result = await session.execute(
                    select(Entity)
                    .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
                    .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
                    .where(Chunk.document_id == document.id)
                )
                entities = entity_result.scalars().all()

            chunk_texts = []
            for chunk in chunks:
                chunk_text = (
                    str(chunk.text) if hasattr(chunk.text, "value") else chunk.text
                )
                chunk_texts.append(chunk_text)

            entity_texts = []
            for entity in entities:
                entity_name = (
                    entity.name if isinstance(entity.name, str) else str(entity.name)
                )
                entity_desc = (
                    entity.description
                    if isinstance(entity.description, str)
                    else str(entity.description or "")
                )
                entity_texts.append(f"{entity_name}. {entity_desc}")

            # Use DocumentPipelineService's embedding client for chunk embeddings
            if chunk_texts:
                embedding_client = self._document_service._embedding_client
                chunk_embeddings = await embedding_client.embed_batch(chunk_texts)
                chunk_updates = [
                    (str(chunk.id), embedding)
                    for chunk, embedding in zip(chunks, chunk_embeddings)
                    if embedding
                ]
                await self._vector_store.update_chunk_embeddings_batch(chunk_updates)

            # Use DocumentPipelineService's embedding client for entity embeddings
            if entity_texts:
                embedding_client = self._document_service._embedding_client
                entity_embeddings = await embedding_client.embed_batch(entity_texts)
                entity_updates = [
                    (str(entity.id), embedding)
                    for entity, embedding in zip(entities, entity_embeddings)
                    if embedding
                ]
                await self._vector_store.update_entity_embeddings_batch(entity_updates)

            obs.log_metric(
                "entities_embedded", len(entities), document_id=str(document.id)
            )
            obs.log_metric("chunks_embedded", len(chunks), document_id=str(document.id))

            async with self._vector_store.get_session() as session:
                doc_to_update = await session.get(Document, document.id)
                if doc_to_update:
                    doc_to_update.status = "embedded"
                    await session.commit()
                    refreshed_doc = await session.get(Document, document.id)
                    if refreshed_doc:
                        document = refreshed_doc

            obs.log_metric("embeddings_created", len(chunks) + len(entities))

            return document

    async def close(self) -> None:
        """Close all components."""
        if self._vector_store:
            await self._vector_store.close()


async def main() -> None:
    """Main entry point."""
    orchestrator = IngestionOrchestrator()

    try:
        await orchestrator.initialize()

        logger.info("Knowledge Base Ingestion System Initialized")
        logger.info("Ready to process documents")

    finally:
        await orchestrator.close()


if __name__ == "__main__":
    asyncio.run(main())
