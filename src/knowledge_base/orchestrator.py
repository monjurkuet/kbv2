"""Main ReAct loop orchestrator."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from knowledge_base.clients import AsyncLLMClient
from knowledge_base.intelligence.v1.adaptive_ingestion_engine import (
    AdaptiveIngestionEngine,
    PipelineRecommendation,
)
from knowledge_base.observability import Observability
from knowledge_base.orchestration.domain_detection_service import DomainDetectionService
from knowledge_base.orchestration.document_pipeline_service import (
    DocumentPipelineService,
)
from knowledge_base.orchestration.entity_pipeline_service import (
    EntityPipelineService,
)
from knowledge_base.orchestration.quality_assurance_service import (
    QualityAssuranceService,
)
from knowledge_base.persistence.v1.schema import (
    Document,
    Entity,
    Edge,
    Chunk,
    ChunkEntity,
)
from knowledge_base.persistence.v1.vector_store import VectorStore
from sqlalchemy import select

logger = logging.getLogger(__name__)


class IngestionOrchestrator:
    """ReAct loop orchestrator for knowledge ingestion - Pure Coordinator Pattern."""

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
        if log_broadcast:
            from knowledge_base.intelligence.v1.extraction_logging import (
                set_websocket_broadcast,
            )

            set_websocket_broadcast(log_broadcast)
        self._observability: Observability | None = None
        self._gateway: AsyncLLMClient | None = None
        self._vector_store: VectorStore | None = None
        self._adaptive_engine: AdaptiveIngestionEngine | None = None
        self._domain_service: DomainDetectionService | None = None
        self._document_service: DocumentPipelineService | None = None
        self._entity_pipeline_service: EntityPipelineService | None = None
        self._quality_assurance_service: QualityAssuranceService | None = None

    async def initialize(self) -> None:
        """Initialize all components - services only, no direct logic."""
        self._observability = Observability()
        self._gateway = AsyncLLMClient()
        self._vector_store = VectorStore()

        await self._vector_store.initialize()

        self._domain_service = DomainDetectionService()
        await self._domain_service.initialize(llm_client=self._gateway)

        self._adaptive_engine = AdaptiveIngestionEngine(gateway=self._gateway)

        self._document_service = DocumentPipelineService()
        await self._document_service.initialize()

        self._entity_pipeline_service = EntityPipelineService()
        await self._entity_pipeline_service.initialize(
            vector_store=self._vector_store,
            gateway=self._gateway,
        )

        self._quality_assurance_service = QualityAssuranceService()
        await self._quality_assurance_service.initialize()

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

    async def process_document(
        self,
        file_path: str | Path,
        document_name: str | None = None,
        domain: str | None = None,
    ) -> Document:
        """Process a document through the full ingestion pipeline - pure delegation.

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
                await self._send_progress(
                    {
                        "step": "started",
                        "status": "started",
                        "message": "Processing document",
                    }
                )

                if not domain:
                    # Read file content for domain detection
                    with open(file_path, "r", encoding="utf-8") as f:
                        content_text = f.read()

                    doc_for_detection = await self._document_service.create_document(
                        file_path=file_path,
                        document_name=document_name,
                        vector_store=self._vector_store,
                    )
                    domain = await self._domain_service.detect_domain(
                        doc_for_detection, content_text
                    )
                    if not domain:
                        domain = "GENERAL"  # Default domain if detection fails

                    logger.info(f"Auto-detected domain: {domain}")

                document = await self._document_service.process(
                    file_path=file_path,
                    document_name=document_name,
                    domain=domain,
                    vector_store=self._vector_store,
                )

                await self._send_progress(
                    {
                        "step": "document_processed",
                        "status": "completed",
                        "message": "Document processed",
                    }
                )

                async with self._vector_store.get_session() as session:
                    chunk_result = await session.execute(
                        select(Chunk).where(Chunk.document_id == document.id).limit(3)
                    )
                    sample_chunks = chunk_result.scalars().all()
                    sample_text = " ".join([chunk.text for chunk in sample_chunks])

                recommendation = await self._adaptive_engine.analyze_document(
                    document_text=sample_text,
                    document_name=document.name,
                    file_size_bytes=sum([len(chunk.text) for chunk in sample_chunks]),
                )

                await self._send_progress(
                    {
                        "step": "extracting",
                        "status": "started",
                        "message": f"Extracting: {recommendation.complexity.value}",
                    }
                )

                async with self._vector_store.get_session() as session:
                    chunk_result = await session.execute(
                        select(Chunk).where(Chunk.document_id == document.id)
                    )
                    chunks = chunk_result.scalars().all()

                entities, edges = await self._entity_pipeline_service.extract(
                    document=document,
                    chunks=list(chunks),
                    domain=domain,
                    use_multi_agent=True,
                    recommendation=recommendation,
                )

                await self._send_progress(
                    {
                        "step": "resolving",
                        "status": "started",
                        "message": "Resolving and clustering",
                    }
                )

                await self._entity_pipeline_service.resolve_and_cluster(
                    document=document,
                    entities=entities,
                    quality_assurance_service=self._quality_assurance_service,
                )

                await self._send_progress(
                    {
                        "step": "entity_complete",
                        "status": "completed",
                        "message": "Entity processing complete",
                    }
                )

                document = await self._finalize_document(document, domain, obs)

                obs.log_event(
                    "document_processing_completed",
                    document_id=str(document.id),
                    document_name=document.name,
                )

                return document

        except Exception as e:
            doc_id = None
            # Check if document was defined in try block
            if locals().get("document") is not None:
                doc_id = getattr(document, "id", None)
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

    async def _finalize_document(
        self,
        document: Document,
        domain: str,
        obs: Observability,
    ) -> Document:
        """Finalize document - update domain and status."""
        await self._send_progress(
            {
                "step": "finalizing",
                "status": "started",
                "message": "Finalizing document",
            }
        )

        async with self._vector_store.get_session() as session:
            doc_to_update = await session.get(Document, document.id)
            if doc_to_update:
                doc_to_update.domain = domain
            entity_result = await session.execute(
                select(Entity)
                .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
                .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
                .where(Chunk.document_id == document.id)
            )
            for entity in entity_result.scalars().all():
                entity.domain = domain
            edge_result = await session.execute(
                select(Edge)
                .join(Entity, Edge.source_id == Entity.id)
                .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
                .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
                .where(Chunk.document_id == document.id)
            )
            for edge in edge_result.scalars().all():
                edge.domain = domain
            await session.commit()

            doc_to_update = await session.get(Document, document.id)
            if doc_to_update:
                doc_to_update.status = "completed"
                await session.commit()
                refreshed_doc = await session.get(Document, document.id)
                if refreshed_doc:
                    document = refreshed_doc

        await self._send_progress(
            {
                "step": "finalizing",
                "status": "completed",
                "message": "Document complete",
            }
        )

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
