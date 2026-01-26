"""Main ReAct loop orchestrator."""

import asyncio
from pathlib import Path
from typing import Callable, Awaitable
from uuid import UUID, uuid4

from sqlalchemy import select

from knowledge_base.common.gateway import GatewayClient
from knowledge_base.common.temporal_utils import TemporalNormalizer
from knowledge_base.ingestion.v1.embedding_client import (
    EmbeddingClient,
)
from knowledge_base.ingestion.v1.gleaning_service import (
    ExtractionResult,
    GleaningService,
)
from knowledge_base.ingestion.v1.partitioning_service import (
    PartitioningService,
)
from knowledge_base.intelligence.v1.clustering_service import (
    ClusteringService,
)
from knowledge_base.intelligence.v1.resolution_agent import (
    ResolutionAgent,
    EntityResolution,
)
from knowledge_base.intelligence.v1.synthesis_agent import (
    SynthesisAgent,
)
from knowledge_base.observability import Observability
from knowledge_base.persistence.v1.schema import (
    Chunk,
    Document,
    DocumentStatus,
    Edge,
    Entity,
    ChunkEntity,
    Community,
)
from knowledge_base.persistence.v1.vector_store import VectorStore


class IngestionOrchestrator:
    """ReAct loop orchestrator for knowledge ingestion."""

    def __init__(
        self,
        progress_callback: Callable[[dict], Awaitable[None] | None] | None = None,
    ) -> None:
        """Initialize orchestrator with all components.

        Args:
            progress_callback: Optional callback function for progress updates.
        """
        self._observability = Observability()
        self._gateway = GatewayClient()
        self._embedding_client = EmbeddingClient()
        self._vector_store = VectorStore()
        self._partitioning_service = PartitioningService()
        self._gleaning_service = GleaningService(self._gateway)
        self._resolution_agent = ResolutionAgent(self._gateway, self._vector_store)
        self._clustering_service = ClusteringService()
        self._synthesis_agent = SynthesisAgent(self._gateway)
        self._temporal_normalizer = TemporalNormalizer()
        self._progress_callback = progress_callback

    async def _emit_progress(self, stage: int, status: str, message: str) -> None:
        """Emit progress update via callback if available.

        Args:
            stage: Stage number (1-9).
            status: Status of the stage ("started" or "completed").
            message: Description of what's happening.
        """
        if self._progress_callback is None:
            return

        progress_data = {
            "type": "progress",
            "stage": stage,
            "status": status,
            "message": message,
        }

        result = self._progress_callback(progress_data)
        if asyncio.iscoroutine(result):
            await result

    async def initialize(self) -> None:
        """Initialize all components."""
        await self._vector_store.initialize()
        await self._vector_store.create_entity_embedding_index()
        await self._vector_store.create_chunk_embedding_index()

    async def process_document(
        self,
        file_path: str | Path,
        document_name: str | None = None,
        domain: str | None = None,
    ) -> Document:
        """Process a document through the full ReAct pipeline.

        Args:
            file_path: Path to document file.
            document_name: Optional document name.
            domain: Optional domain for the document (e.g., technology, healthcare, finance).
                    If not provided, domain will be determined automatically.

        Returns:
            Processed document.
        """
        obs = self._observability

        async with obs.trace_context(
            "document_ingestion",
            file_path=str(file_path),
        ):
            await self._emit_progress(1, "started", "Creating document record")
            document = await self._create_document(file_path, document_name)
            await self._emit_progress(1, "completed", "Document record created")

            try:
                await self._emit_progress(
                    2, "started", "Partitioning document into chunks"
                )
                document = await self._partition_document(document, file_path)
                await self._emit_progress(
                    2, "completed", "Document partitioned into chunks"
                )

                await self._emit_progress(
                    3, "started", "Extracting knowledge (entities and edges)"
                )
                document = await self._extract_knowledge(document)
                await self._emit_progress(
                    3, "completed", "Knowledge extraction complete"
                )

                await self._emit_progress(4, "started", "Generating embeddings")
                document = await self._embed_content(document)
                await self._emit_progress(4, "completed", "Embeddings generated")

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
                # Set domain for document and propagate to entities and edges
                # Use provided domain or determine automatically
                final_domain = (
                    domain if domain is not None else self._determine_domain(document)
                )

                # Update domain for document, entities and edges associated with this document
                async with self._vector_store.get_session() as session:
                    # Update document domain
                    doc_to_update = await session.get(Document, document.id)
                    if doc_to_update:
                        doc_to_update.domain = final_domain

                    # Update entities with the specified domain
                    entity_result = await session.execute(
                        select(Entity)
                        .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
                        .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
                        .where(Chunk.document_id == document.id)
                    )
                    entities = entity_result.scalars().all()
                    for entity in entities:
                        entity.domain = final_domain

                    # Update edges with the specified domain
                    edge_result = await session.execute(
                        select(Edge)
                        .join(Entity, Edge.source_id == Entity.id)
                        .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
                        .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
                        .where(Chunk.document_id == document.id)
                    )
                    edges = edge_result.scalars().all()
                    for edge in edges:
                        edge.domain = final_domain

                    await session.commit()
                await self._emit_progress(8, "completed", "Domain updated successfully")

                await self._emit_progress(9, "started", "Finalizing document status")
                # Update document status
                async with self._vector_store.get_session() as session:
                    doc_to_update = await session.get(Document, document.id)
                    if doc_to_update:
                        doc_to_update.status = DocumentStatus.COMPLETED
                    await session.commit()
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
                # Update document status to failed
                async with self._vector_store.get_session() as session:
                    doc_to_update = await session.get(Document, document.id)
                    if doc_to_update:
                        doc_to_update.status = DocumentStatus.FAILED
                    await session.commit()

                obs.log_event(
                    "document_processing_failed",
                    document_id=str(document.id),
                    error=str(e),
                    level="error",
                )

                raise

    async def _create_document(
        self,
        file_path: str | Path,
        document_name: str | None,
    ) -> Document:
        """Create document record.

        Args:
            file_path: Path to file.
            document_name: Document name.

        Returns:
            Created document.
        """
        path = Path(file_path)
        name = document_name or path.name

        document = Document(
            id=uuid4(),
            name=name,
            source_uri=str(path.absolute()),
            mime_type=self._get_mime_type(path),
            status=DocumentStatus.PENDING,
        )

        async with self._vector_store.get_session() as session:
            session.add(document)
            await session.commit()
            document = await session.get(Document, document.id)

        return document

    def _determine_domain(self, document: Document) -> str:
        """Determine domain for a document based on its content or metadata.

        Args:
            document: The document to determine domain for.

        Returns:
            Domain string (e.g., "technology", "healthcare", "finance").
        """
        # Simple domain determination based on document name and metadata
        # In a real implementation, this would use more sophisticated analysis
        if not document.doc_metadata:
            return "general"

        # Check for domain in metadata
        if "domain" in document.doc_metadata:
            return str(document.doc_metadata["domain"])

        # Fallback to name-based heuristics
        name_lower = document.name.lower()
        if any(term in name_lower for term in ["tech", "software", "code", "api"]):
            return "technology"
        elif any(
            term in name_lower for term in ["health", "medical", "patient", "doctor"]
        ):
            return "healthcare"
        elif any(
            term in name_lower for term in ["finance", "bank", "money", "investment"]
        ):
            return "finance"
        else:
            return "general"

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
            chunks = await self._partitioning_service.partition_and_chunk(file_path)

            async with self._vector_store.get_session() as session:
                for idx, chunk_data in enumerate(chunks):
                    chunk = Chunk(
                        id=uuid4(),
                        document_id=document.id,
                        text=chunk_data.text,
                        chunk_index=idx,
                        page_number=chunk_data.page_number,
                        token_count=len(chunk_data.text.split()),
                        metadata=chunk_data.metadata,
                    )
                    session.add(chunk)

                document = await session.merge(document)
                document.status = DocumentStatus.PARTITIONED
                await session.commit()
                document = await session.get(Document, document.id)

            obs.log_metric("chunks_created", len(chunks), document_id=str(document.id))

            return document

    async def _extract_knowledge(self, document: Document) -> Document:
        """Extract entities, edges, and temporal claims.

        Args:
            document: Document record.

        Returns:
            Updated document.
        """
        obs = self._observability

        async with obs.trace_context("extract_knowledge", document_id=str(document.id)):
            async with self._vector_store.get_session() as session:
                result = await session.execute(
                    select(Chunk)
                    .where(Chunk.document_id == document.id)
                    .order_by(Chunk.chunk_index)
                )
                chunks = result.scalars().all()

            all_entities: list[Entity] = []
            all_edges: list[Edge] = []
            chunk_entity_links: list[tuple[UUID, UUID]] = []

            for chunk in chunks:
                extraction = await self._gleaning_service.extract(chunk.text)

                entities = self._create_entities_from_extraction(extraction, chunk.id)
                edges = self._create_edges_from_extraction(extraction, entities)

                all_entities.extend(entities)
                all_edges.extend(edges)

                for entity in entities:
                    chunk_entity_links.append((chunk.id, entity.id))

            async with self._vector_store.get_session() as session:
                from sqlalchemy.dialects.postgresql import insert as pg_insert
                from uuid import uuid4
                
                # Use on_conflict_do_nothing to avoid the issue entirely
                processed_uris = set()  # Track processed entities per document
                for entity in all_entities:
                    # Skip if we already processed this URI in this document
                    if entity.uri in processed_uris:
                        continue
                    processed_uris.add(entity.uri)
                    
                    stmt = pg_insert(Entity).values(
                        id=entity.id,
                        name=entity.name,
                        entity_type=entity.entity_type,
                        description=entity.description,
                        properties=entity.properties,
                        confidence=entity.confidence,
                        uri=entity.uri,
                        source_text=entity.source_text
                    ).on_conflict_do_nothing()
                    await session.execute(stmt)

                # Insert edges with upsert to avoid duplicates and missing entity errors
                for edge in all_edges:
                    try:
                        edge_stmt = pg_insert(Edge).values(
                            id=edge.id,
                            source_id=edge.source_id,
                            target_id=edge.target_id,
                            edge_type=edge.edge_type,
                            properties=edge.properties,
                            confidence=edge.confidence,
                            source_text=edge.source_text
                        ).on_conflict_do_nothing()
                        await session.execute(edge_stmt)
                    except Exception as e:
                        # Log but continue - edge might reference non-existent entity
                        # due to deduplication
                        logger.debug(f"Skipping edge {edge.id}: {e}")
                        continue

                await session.commit()

                # Link chunks to entities (avoid duplicates)
                for chunk_id, entity_id in chunk_entity_links:
                    link_stmt = pg_insert(ChunkEntity).values(
                        id=uuid4(),
                        chunk_id=chunk_id,
                        entity_id=entity_id
                    ).on_conflict_do_nothing()
                    await session.execute(link_stmt)

                await session.commit()

            document.status = DocumentStatus.EXTRACTED
            await self._update_document(document)

            obs.log_metric(
                "entities_extracted", len(all_entities), document_id=str(document.id)
            )
            obs.log_metric(
                "edges_extracted", len(all_edges), document_id=str(document.id)
            )

            return document

    async def _embed_content(self, document: Document) -> Document:
        """Generate embeddings for entities and chunks.

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

            for chunk in chunks:
                embedding = await self._embedding_client.embed_text(chunk.text)
                await self._vector_store.update_chunk_embedding(
                    str(chunk.id), embedding
                )

            for entity in entities:
                text = f"{entity.name}. {entity.description or ''}"
                embedding = await self._embedding_client.embed_text(text)
                await self._vector_store.update_entity_embedding(
                    str(entity.id), embedding
                )

            document.status = DocumentStatus.EMBEDDED
            await self._update_document(document)

            obs.log_metric(
                "entities_embedded", len(entities), document_id=str(document.id)
            )
            obs.log_metric("chunks_embedded", len(chunks), document_id=str(document.id))

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

            for entity in entities:
                similar = await self._vector_store.search_similar_entities(
                    entity.embedding or [],
                    limit=5,
                    similarity_threshold=0.85,
                )

                candidates = [
                    e
                    for e in entities
                    if e.id != entity.id and str(e.id) in [s["id"] for s in similar]
                ]

                if candidates:
                    resolution = await self._resolution_agent.resolve_entity(
                        entity,
                        candidates,
                        "",
                    )

                    if resolution.merged_entity_ids:
                        await self._merge_entities(resolution)

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

            hierarchy = None

            if entities and edges:
                hierarchy = await self._clustering_service.build_hierarchy(
                    entities, edges
                )

                async with self._vector_store.get_session() as session:
                    # Persist all communities from the hierarchy
                    for community_id, community in hierarchy.communities.items():
                        existing = (
                            await session.execute(
                                select(Community).where(Community.id == community_id)
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

            obs.log_metric(
                "communities_created",
                len(hierarchy.community_ids) if hierarchy else 0,
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
                    entities,
                    edges,
                )

                async with self._vector_store.get_session() as session:
                    for community in communities:
                        if community.id in reports:
                            report = reports[community.id]
                            community.summary = report.summary
                            await session.commit()

            return document

    async def _merge_entities(self, resolution: EntityResolution) -> None:
        """Merge entities based on resolution.

        Args:
            resolution: Entity resolution result.
        """
        async with self._vector_store.get_session() as session:
            for merged_id in resolution.merged_entity_ids:
                edges = await session.execute(
                    select(Edge).where(
                        (Edge.source_id == merged_id) | (Edge.target_id == merged_id)
                    )
                )

                for edge in edges.scalars().all():
                    if edge.source_id == merged_id:
                        edge.source_id = resolution.entity_id
                    else:
                        edge.target_id = resolution.entity_id

                await session.delete(await session.get(Entity, merged_id))

            await session.commit()

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

    def _create_entities_from_extraction(
        self,
        extraction: ExtractionResult,
        chunk_id: UUID,
    ) -> list[Entity]:
        """Create entities from extraction result.

        Args:
            extraction: Extraction result.
            chunk_id: Source chunk ID.

        Returns:
            List of entities.
        """
        entities = []
        for entity in extraction.entities:
            # Generate a stable URI for the entity to follow knowledge graph best practices
            import re
            from urllib.parse import quote

            # Create a normalized name for the URI
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
                uri=uri,  # Populate the enhanced URI field following RDF patterns
            )
            entities.append(db_entity)

        return entities

    def _create_edges_from_extraction(
        self,
        extraction: ExtractionResult,
        entities: list[Entity],
    ) -> list[Edge]:
        """Create edges from extraction result.

        Args:
            extraction: Extraction result.
            entities: List of created entities.

        Returns:
            List of edges.
        """
        entity_map = {e.name: e.id for e in entities}
        from knowledge_base.persistence.v1.schema import EdgeType

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

        # Find Project Nova entity
        project_nova = None
        for entity in entities:
            if "project nova" in entity.name.lower():
                project_nova = entity
                break

        # Find person entities (Elena Vance variants)
        person_entities = [
            e
            for e in entities
            if e.entity_type in ("Person", "PERSON") and "vance" in e.name.lower()
        ]

        # Process temporal claims to create timeline edges
        # Map dates to statuses based on all claims
        date_to_status = {}  # Maps date -> status

        # First pass: collect all dates
        date_claims = {}  # Maps date -> list of claims
        for temporal_claim in extraction.temporal_claims:
            date_str = None
            if temporal_claim.iso8601_date:
                date_str = temporal_claim.iso8601_date.split("T")[0]  # YYYY-MM-DD
            elif temporal_claim.start_date:
                date_str = temporal_claim.start_date.strftime("%Y-%m-%d")

            if date_str:
                if date_str not in date_claims:
                    date_claims[date_str] = []
                date_claims[date_str].append(temporal_claim)

        # Second pass: determine status for each date
        for date_str, claims in date_claims.items():
            status = None
            for claim in claims:
                claim_lower = claim.text.lower()
                if "active" in claim_lower:
                    status = "Active"
                elif "failed" in claim_lower:
                    status = "Failed"
                elif "success" in claim_lower or "completed" in claim_lower:
                    status = "Success"
            if status:
                date_to_status[date_str] = status

        # Create temporal edges for each date-status pair
        if project_nova:
            # Use person entities if available, otherwise use any entity that could be a person
            source_entities = (
                person_entities
                if person_entities
                else [e for e in entities if e.entity_type in ("Person", "PERSON")]
            )

            if source_entities:
                # Use the first person entity as the source for all temporal edges
                # to avoid creating duplicate edges for the same date
                source_entity = source_entities[0]

                for date_str, status in date_to_status.items():
                    # Don't create self-loops
                    if source_entity.id != project_nova.id:
                        properties = {
                            "status": status,
                            "date": date_str,
                        }

                        edge = Edge(
                            id=uuid4(),
                            source_id=source_entity.id,
                            target_id=project_nova.id,
                            edge_type=EdgeType.WORKS_FOR,
                            properties=properties,
                            confidence=1.0,
                        )
                        edges.append(edge)

        return edges

    async def close(self) -> None:
        """Close all components."""
        await self._gateway.close()
        await self._vector_store.close()


async def main() -> None:
    """Main entry point."""
    orchestrator = IngestionOrchestrator()

    try:
        await orchestrator.initialize()

        print("Knowledge Base Ingestion System Initialized")
        print("Ready to process documents")

    finally:
        await orchestrator.close()


if __name__ == "__main__":
    asyncio.run(main())
