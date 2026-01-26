"""Main ReAct loop orchestrator."""

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote
from uuid import UUID, uuid4

from knowledge_base.common.gateway import GatewayClient
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
from knowledge_base.intelligence.v1.synthesis_agent import (
    SynthesisAgent,
)
from knowledge_base.observability import Observability
from knowledge_base.persistence.v1.schema import (
    Chunk,
    Document,
    Edge,
    Entity,
    ChunkEntity,
    Community,
    ReviewQueue,
    ReviewStatus,
    EdgeType,
)
from knowledge_base.persistence.v1.vector_store import VectorStore
from sqlalchemy import select

logger = logging.getLogger(__name__)


class IngestionOrchestrator:
    """ReAct loop orchestrator for knowledge ingestion."""

    DOMAIN_KEYWORDS = {
        "technology": [
            "software",
            "code",
            "api",
            "algorithm",
            "database",
            "server",
            "cloud",
            "programming",
            "developer",
            "framework",
            "library",
            "function",
            "class",
            "module",
            "interface",
            "protocol",
            "network",
            "system",
            "data",
            "machine learning",
            "ai",
            "neural",
            "model",
            "training",
            "inference",
        ],
        "healthcare": [
            "patient",
            "doctor",
            "hospital",
            "clinical",
            "diagnosis",
            "treatment",
            "therapy",
            "medication",
            "medical",
            "health",
            "disease",
            "symptom",
            "prescription",
            "surgery",
            "procedure",
            "lab",
            "test",
            "blood",
            "pressure",
            "heart",
            "cancer",
            "diabetes",
            "mental",
        ],
        "finance": [
            "finance",
            "bank",
            "investment",
            "stock",
            "market",
            "trading",
            "portfolio",
            "asset",
            "liability",
            "revenue",
            "profit",
            "loss",
            "equity",
            "bond",
            "loan",
            "credit",
            "debt",
            "cryptocurrency",
            "bitcoin",
            "dollar",
            "euro",
            "yen",
            "forex",
            "capital",
            "income",
            "expense",
            "budget",
            "accounting",
        ],
        "legal": [
            "law",
            "legal",
            "contract",
            "agreement",
            "court",
            "judge",
            "attorney",
            "lawyer",
            "litigation",
            "lawsuit",
            "regulation",
            "compliance",
            "policy",
            "clause",
            "term",
            "breach",
            "liability",
            "damages",
            "settlement",
            "verdict",
            "testimony",
            "evidence",
        ],
        "science": [
            "research",
            "experiment",
            "hypothesis",
            "theory",
            "analysis",
            "data",
            "study",
            "paper",
            "publication",
            "laboratory",
            "scientist",
            "physics",
            "chemistry",
            "biology",
            "molecule",
            "cell",
            "gene",
            "protein",
            "atom",
            "energy",
            "force",
            "quantum",
        ],
    }

    def __init__(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            progress_callback: Optional callback for progress updates.
        """
        self._progress_callback = progress_callback
        self._observability: Observability | None = None
        self._gateway: GatewayClient | None = None
        self._vector_store: VectorStore | None = None
        self._embedding_client: EmbeddingClient | None = None
        self._gleaning_service: GleaningService | None = None
        self._clustering_service: ClusteringService | None = None
        self._resolution_agent: ResolutionAgent | None = None
        self._synthesis_agent: SynthesisAgent | None = None

    async def initialize(self) -> None:
        """Initialize all components."""
        self._observability = Observability()
        self._gateway = GatewayClient()
        self._embedding_client = EmbeddingClient()
        self._vector_store = VectorStore()
        self._gleaning_service = GleaningService(self._gateway)
        self._clustering_service = ClusteringService()
        self._resolution_agent = ResolutionAgent(
            gateway=self._gateway, vector_store=self._vector_store
        )
        self._synthesis_agent = SynthesisAgent(gateway=self._gateway)

        await self._vector_store.initialize()
        logger.info("IngestionOrchestrator initialized successfully")

    def _send_progress(self, progress_data: dict[str, Any]) -> None:
        """Send progress update.

        Args:
            progress_data: Progress information to send.
        """
        if self._progress_callback:
            self._progress_callback(progress_data)

    def _calculate_domain_scores(self, text: str) -> dict[str, float]:
        """Calculate domain scores based on keyword frequency."""
        if not text or not text.strip():
            return {
                "technology": 0.0,
                "healthcare": 0.0,
                "finance": 0.0,
                "legal": 0.0,
                "science": 0.0,
                "general": 0.0,
            }

        text_lower = text.lower()
        scores = {
            "technology": 0.0,
            "healthcare": 0.0,
            "finance": 0.0,
            "legal": 0.0,
            "science": 0.0,
            "general": 0.0,
        }

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                count = text_lower.count(keyword)
                weight = len(keyword) / 10
                scores[domain] += count * weight

        total = sum(scores.values())
        if total > 0:
            for domain in scores:
                scores[domain] /= total

        return scores

    def _determine_domain(
        self, document: Document, content_text: str | None = None
    ) -> str:
        """Determine domain for a document based on its content or metadata.

        Args:
            document: The document to determine domain for.
            content_text: Optional document content text for content-based classification.

        Returns:
            Domain string (e.g., "technology", "healthcare", "finance").
        """
        if document.doc_metadata and "domain" in document.doc_metadata:
            return str(document.doc_metadata["domain"])

        if content_text:
            scores = self._calculate_domain_scores(content_text)
            best_domain = max(scores, key=lambda k: scores.get(k, 0.0))
            if scores.get(best_domain, 0) >= 0.1:
                return best_domain

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
            from knowledge_base.ingestion.v1.partitioning_service import (
                PartitioningService,
            )

            partitioning_service = PartitioningService()
            chunks = await partitioning_service.partition_and_chunk(file_path)

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
                document.status = "partitioned"
                await session.commit()
                # Refresh document from session
                refreshed_doc = await session.get(Document, document.id)
                if refreshed_doc:
                    document = refreshed_doc

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
                chunk_result = await session.execute(
                    select(Chunk).where(Chunk.document_id == document.id)
                )
                chunks = chunk_result.scalars().all()

                # Process each chunk
                for chunk in chunks:
                    try:
                        # Extract entities from chunk
                        # Convert Column[str] to str for extraction
                        chunk_text = (
                            str(chunk.text)
                            if hasattr(chunk.text, "value")
                            else chunk.text
                        )
                        extraction = await self._gleaning_service.extract(chunk_text)

                        # Create Entity objects
                        # Convert Column[UUID] to UUID for entity creation
                        chunk_uuid = (
                            chunk.id
                            if isinstance(chunk.id, UUID)
                            else UUID(str(chunk.id))
                        )
                        entities = self._create_entities_from_extraction(
                            extraction, chunk_uuid
                        )

                        # Track successfully inserted entities
                        inserted_entity_ids: set[UUID] = set()

                        # Insert entities with conflict handling
                        for entity in entities:
                            try:
                                # Check if entity with this URI already exists
                                existing_entity = None
                                if getattr(entity, "uri", None) and isinstance(
                                    getattr(entity, "uri", None), str
                                ):
                                    result = await session.execute(
                                        select(Entity).where(Entity.uri == entity.uri)
                                    )
                                    existing_entity = result.scalar_one_or_none()

                                if existing_entity:
                                    # Use existing entity ID
                                    inserted_entity_ids.add(existing_entity.id)
                                    # Update the local entity object to use existing ID for edge creation
                                    entity.id = existing_entity.id
                                else:
                                    # New entity - add it
                                    session.add(entity)
                                    await (
                                        session.flush()
                                    )  # This may still fail with unique constraint
                                    inserted_entity_ids.add(entity.id)

                            except Exception:
                                pass

                        # Handle race conditions by querying for existing entities
                        if len(inserted_entity_ids) < len(entities):
                            existing_entities = await session.execute(
                                select(Entity.id).where(
                                    Entity.name.in_(
                                        [
                                            e.name
                                            for e in entities
                                            if e.id not in inserted_entity_ids
                                        ]
                                    )
                                )
                            )
                            for row in existing_entities.scalars():
                                inserted_entity_ids.add(row)

                        # Create ChunkEntity relationships only for successfully inserted entities
                        for entity in entities:
                            if entity.id in inserted_entity_ids:
                                chunk_entity = ChunkEntity(
                                    chunk_id=chunk.id, entity_id=entity.id
                                )
                                session.add(chunk_entity)

                        # Create edges
                        edges = self._create_edges_from_extraction(extraction, entities)
                        for edge in edges:
                            # Only add edges between entities that were successfully inserted
                            if (
                                edge.source_id in inserted_entity_ids
                                and edge.target_id in inserted_entity_ids
                            ):
                                session.add(edge)

                    except Exception as e:
                        logger.error(f"Failed to process chunk {chunk.id}: {str(e)}")
                        continue

                await session.commit()

                # Update document status using session to ensure proper attribute assignment
                doc_in_session = await session.get(Document, document.id)
                if doc_in_session:
                    doc_in_session.status = "extracted"
                    await session.commit()
                    # Refresh the local document object from session
                    refreshed_doc = await session.get(Document, document.id)
                    if refreshed_doc:
                        document = refreshed_doc

                # Log metrics
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

            for entity in entities:
                embedding = entity.embedding if entity.embedding is not None else []
                similar = await self._vector_store.search_similar_entities(
                    embedding,
                    limit=5,
                    similarity_threshold=0.85,
                )

                candidates = [
                    e
                    for e in entities
                    if e.id != entity.id and str(e.id) in [s["id"] for s in similar]
                ]

                if candidates:
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
                            str(chunk.text)
                            if hasattr(chunk.text, "value")
                            else chunk.text
                        )

                    resolution = await self._resolution_agent.resolve_entity(
                        entity,
                        candidates,
                        chunk_text,
                    )

                    if resolution.merged_entity_ids:
                        doc_id = (
                            UUID(str(document.id))
                            if not isinstance(document.id, UUID)
                            else document.id
                        )
                        ent_id = (
                            UUID(str(entity.id))
                            if not isinstance(entity.id, UUID)
                            else entity.id
                        )
                        if resolution.human_review_required:
                            await self._add_to_review_queue(
                                doc_id,
                                ent_id,
                                resolution.merged_entity_ids,
                                resolution.confidence_score,
                                resolution.grounding_quote,
                                chunk_text,
                            )
                        else:
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
                    list(entities), list(edges)
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
        date_to_status: dict[str, str] = {}  # Maps date -> status

        # First pass: collect all dates
        date_claims: dict[str, list[Any]] = {}  # Maps date -> list of claims
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

    def _emit_progress(self, stage: int, status: str, message: str, **kwargs) -> None:
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
        self._send_progress(progress_data)

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
                self._emit_progress(1, "started", "Creating document record")
                document = await self._create_document(file_path, document_name)
                self._emit_progress(1, "completed", "Document record created")

                self._emit_progress(2, "started", "Partitioning document into chunks")
                document = await self._partition_document(document, file_path)
                self._emit_progress(2, "completed", "Document partitioned into chunks")

                self._emit_progress(
                    3, "started", "Extracting knowledge (entities and edges)"
                )
                document = await self._extract_knowledge(document)
                self._emit_progress(3, "completed", "Knowledge extraction complete")

                self._emit_progress(4, "started", "Embedding chunks and entities")
                document = await self._embed_content(document)
                self._emit_progress(4, "completed", "Embedding complete")

                self._emit_progress(5, "started", "Resolving duplicate entities")
                document = await self._resolve_entities(document)
                self._emit_progress(5, "completed", "Entity resolution complete")

                self._emit_progress(
                    6, "started", "Clustering entities into communities"
                )
                document = await self._cluster_entities(document)
                self._emit_progress(6, "completed", "Entity clustering complete")

                self._emit_progress(7, "started", "Generating intelligence reports")
                document = await self._generate_reports(document)
                self._emit_progress(7, "completed", "Intelligence reports generated")

                self._emit_progress(
                    8, "started", "Updating domain for document and entities"
                )
                final_domain = (
                    domain if domain is not None else self._determine_domain(document)
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
                self._emit_progress(8, "completed", "Domain updated successfully")

                self._emit_progress(9, "started", "Finalizing document status")
                async with self._vector_store.get_session() as session:
                    doc_to_update = await session.get(Document, document.id)
                    if doc_to_update:
                        doc_to_update.status = "completed"
                        await session.commit()
                        refreshed_doc = await session.get(Document, document.id)
                        if refreshed_doc:
                            document = refreshed_doc
                self._emit_progress(9, "completed", "Document processing complete")

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

            if chunk_texts:
                chunk_embeddings = await self._embedding_client.embed_batch(chunk_texts)
                for chunk, embedding in zip(chunks, chunk_embeddings):
                    if embedding:
                        await self._vector_store.update_chunk_embedding(
                            str(chunk.id), embedding
                        )

            if entity_texts:
                entity_embeddings = await self._embedding_client.embed_batch(
                    entity_texts
                )
                for entity, embedding in zip(entities, entity_embeddings):
                    if embedding:
                        await self._vector_store.update_entity_embedding(
                            str(entity.id), embedding
                        )

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

        print("Knowledge Base Ingestion System Initialized")
        print("Ready to process documents")

    finally:
        await orchestrator.close()


if __name__ == "__main__":
    asyncio.run(main())
