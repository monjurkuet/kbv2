"""Enhanced Ingestion Orchestrator with Self-Improvement.

This module extends the base orchestrator to integrate:
- Experience Bank for few-shot prompting
- Prompt Evolution for automated prompt optimization
"""

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from knowledge_base.intelligence.v1.self_improvement import (
    ExperienceBank,
    ExperienceBankConfig,
    ExperienceBankMiddleware,
    PromptEvolutionEngine,
    PromptEvolutionConfig,
    OntologyValidator,
)
from knowledge_base.orchestrator import IngestionOrchestrator
from knowledge_base.persistence.v1.schema import Chunk
from sqlalchemy import select

logger = logging.getLogger(__name__)


class SelfImprovingOrchestrator(IngestionOrchestrator):
    """Extended orchestrator with self-improvement capabilities.

    Adds:
    - Experience Bank integration for few-shot prompting
    - Prompt Evolution for automated prompt optimization
    - Automatic storage of high-quality extractions
    """

    def __init__(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        log_broadcast: Optional[Callable[[str], Any]] = None,
        enable_experience_bank: bool = True,
        enable_prompt_evolution: bool = True,
        enable_ontology_validation: bool = True,
    ) -> None:
        """Initialize self-improving orchestrator.

        Args:
            progress_callback: Optional callback for progress updates
            log_broadcast: Optional callback for WebSocket logging
            enable_experience_bank: Whether to enable experience bank
            enable_prompt_evolution: Whether to enable prompt evolution
            enable_ontology_validation: Whether to enable ontology validation
        """
        super().__init__(progress_callback, log_broadcast)

        self._enable_experience_bank = enable_experience_bank
        self._enable_prompt_evolution = enable_prompt_evolution
        self._enable_ontology_validation = enable_ontology_validation

        self._experience_bank: Optional[ExperienceBank] = None
        self._experience_middleware: Optional[ExperienceBankMiddleware] = None
        self._prompt_evolution: Optional[PromptEvolutionEngine] = None
        self._ontology_validator: Optional[OntologyValidator] = None

    async def initialize(self) -> None:
        """Initialize base components and self-improvement features."""
        # Initialize base orchestrator
        await super().initialize()

        # Initialize Experience Bank
        if self._enable_experience_bank:
            await self._initialize_experience_bank()

        # Initialize Prompt Evolution
        if self._enable_prompt_evolution:
            await self._initialize_prompt_evolution()

        # Initialize Ontology Validator
        if self._enable_ontology_validation:
            await self._initialize_ontology_validator()

        logger.info("SelfImprovingOrchestrator initialized successfully")

    async def _initialize_experience_bank(self) -> None:
        """Initialize the experience bank."""
        try:
            # Create config
            config = ExperienceBankConfig(
                min_quality_threshold=0.85,
                max_storage_size=10000,
                similarity_top_k=3,
                enable_pattern_extraction=True,
            )

            # Get database session
            async with self._vector_store.get_session() as session:
                self._experience_bank = ExperienceBank(
                    session=session,
                    vector_store=self._vector_store,
                    config=config,
                )
                self._experience_middleware = ExperienceBankMiddleware(
                    self._experience_bank
                )

            logger.info("Experience Bank initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Experience Bank: {e}")
            self._enable_experience_bank = False

    async def _initialize_prompt_evolution(self) -> None:
        """Initialize prompt evolution engine."""
        try:
            config = PromptEvolutionConfig()

            self._prompt_evolution = PromptEvolutionEngine(
                gateway=self._gateway,
                config=config,
            )

            # Initialize crypto domains
            for domain in config.crypto_domains:
                await self._prompt_evolution.initialize_domain(domain)

            logger.info(
                f"Prompt Evolution initialized for {len(config.crypto_domains)} domains"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Prompt Evolution: {e}")
            self._enable_prompt_evolution = False

    async def _initialize_ontology_validator(self) -> None:
        """Initialize the ontology validator."""
        try:
            self._ontology_validator = OntologyValidator()
            logger.info("Ontology Validator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Ontology Validator: {e}")
            self._enable_ontology_validation = False

    async def process_document(
        self,
        file_path: str | Path,
        document_name: str | None = None,
        domain: str | None = None,
    ) -> Any:
        """Process document with self-improvement features.

        Args:
            file_path: Path to document file
            document_name: Optional document name
            domain: Optional domain for the document

        Returns:
            Processed document
        """
        # Call base process_document but inject our enhancements
        # We need to override entity extraction to use Experience Bank

        return await self._process_with_self_improvement(
            file_path=file_path,
            document_name=document_name,
            domain=domain,
        )

    async def _process_with_self_improvement(
        self,
        file_path: str | Path,
        document_name: str | None = None,
        domain: str | None = None,
    ) -> Any:
        """Process document with Experience Bank integration."""
        obs = self._observability

        try:
            # Domain detection (from base)
            if not domain:
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
                    domain = "GENERAL"

                logger.info(f"Auto-detected domain: {domain}")

            # Document processing (from base)
            document = await self._document_service.process(
                file_path=file_path,
                document_name=document_name,
                domain=domain,
                vector_store=self._vector_store,
            )

            # Get chunks
            async with self._vector_store.get_session() as session:
                chunk_result = await session.execute(
                    select(Chunk).where(Chunk.document_id == document.id)
                )
                chunks = chunk_result.scalars().all()

            # Get adaptive recommendation
            sample_text = " ".join([chunk.text for chunk in chunks[:3]])
            recommendation = await self._adaptive_engine.analyze_document(
                document_text=sample_text,
                document_name=document.name,
                file_size_bytes=sum([len(chunk.text) for chunk in chunks]),
            )

            # ENHANCEMENT: Enrich extraction with Experience Bank examples
            enriched_chunks = await self._enrich_chunks_with_examples(
                chunks=list(chunks),
                domain=domain,
            )

            # Entity extraction (with potential prompt evolution)
            entities, edges = await self._extract_with_self_improvement(
                document=document,
                chunks=enriched_chunks,
                domain=domain,
                recommendation=recommendation,
            )

            # ENHANCEMENT: Validate entities against ontology
            if self._enable_ontology_validation and self._ontology_validator:
                entities, edges = await self._validate_extraction(
                    entities=entities,
                    edges=edges,
                    domain=domain,
                )

            # ENHANCEMENT: Store high-quality extractions
            await self._store_extraction_experience(
                chunks=list(chunks),
                entities=entities,
                edges=edges,
                domain=domain,
            )

            # Continue with base processing
            await self._entity_pipeline_service.resolve_and_cluster(
                document=document,
                entities=entities,
                quality_assurance_service=self._quality_assurance_service,
            )

            # Finalize
            document = await self._finalize_document(document, domain, obs)

            obs.log_event(
                "document_processing_completed_with_self_improvement",
                document_id=str(document.id),
                document_name=document.name,
                domain=domain,
                entity_count=len(entities),
                edge_count=len(edges),
            )

            return document

        except Exception as e:
            # Error handling from base
            raise

    async def _enrich_chunks_with_examples(
        self,
        chunks: list,
        domain: str,
    ) -> list:
        """Enrich chunks with Experience Bank examples.

        Args:
            chunks: Document chunks
            domain: Document domain

        Returns:
            Enriched chunks (or original if enrichment fails)
        """
        if not self._enable_experience_bank or not self._experience_middleware:
            return chunks

        try:
            # Retrieve examples for this domain
            examples = await self._experience_bank.retrieve_similar_examples(
                text=chunks[0].text if chunks else "",
                domain=domain,
                k=3,
            )

            if examples:
                logger.info(f"Retrieved {len(examples)} examples for domain {domain}")
                # Store examples in chunk metadata for extraction to use
                for chunk in chunks:
                    if not hasattr(chunk, "metadata"):
                        chunk.metadata = {}
                    chunk.metadata["experience_examples"] = [
                        ex.to_few_shot_format() for ex in examples
                    ]

            return chunks

        except Exception as e:
            logger.warning(f"Failed to enrich chunks with examples: {e}")
            return chunks

    async def _extract_with_self_improvement(
        self,
        document: Any,
        chunks: list,
        domain: str,
        recommendation: Any,
    ) -> tuple:
        """Extract entities with self-improvement features.

        Args:
            document: Document being processed
            chunks: Document chunks
            domain: Document domain
            recommendation: Pipeline recommendation

        Returns:
            Tuple of (entities, edges)
        """
        # Check if we should use evolved prompt
        if self._enable_prompt_evolution and self._prompt_evolution:
            try:
                # Get best prompt for domain
                evolved_prompt = await self._prompt_evolution.get_best_prompt(domain)

                # Store evolved prompt in document metadata for extraction to use
                if not hasattr(document, "metadata"):
                    document.metadata = {}
                document.metadata["evolved_prompt"] = evolved_prompt

                logger.info(f"Using evolved prompt for domain {domain}")

            except Exception as e:
                logger.warning(f"Failed to get evolved prompt: {e}")

        # Call base extraction
        entities, edges = await self._entity_pipeline_service.extract(
            document=document,
            chunks=chunks,
            domain=domain,
            use_multi_agent=True,
            recommendation=recommendation,
        )

        return entities, edges

    async def _validate_extraction(
        self,
        entities: list,
        edges: list,
        domain: str,
    ) -> tuple:
        """Validate extraction against ontology rules.

        Args:
            entities: Extracted entities
            edges: Extracted edges
            domain: Document domain

        Returns:
            Tuple of (validated_entities, validated_edges)
        """
        try:
            report = await self._ontology_validator.validate_extraction(
                entities=entities,
                relationships=edges,
            )

            # Log validation results
            if report.violations:
                logger.info(
                    f"Ontology validation: {len(report.violations)} violations found, "
                    f"score: {report.overall_score:.2f}"
                )
                for violation in report.violations[:5]:  # Log first 5
                    logger.debug(f"  - {violation.violation_type}: {violation.message}")
            else:
                logger.info(
                    f"Ontology validation: All entities valid (score: {report.overall_score:.2f})"
                )

            # Filter out entities with critical violations
            if report.overall_score < 0.5:
                logger.warning(
                    f"Low validation score ({report.overall_score:.2f}), reviewing entities"
                )

            return entities, edges

        except Exception as e:
            logger.warning(f"Ontology validation failed: {e}")
            return entities, edges

    async def _store_extraction_experience(
        self,
        chunks: list,
        entities: list,
        edges: list,
        domain: str,
    ) -> None:
        """Store high-quality extractions in Experience Bank.

        Args:
            chunks: Document chunks
            entities: Extracted entities
            edges: Extracted edges
            domain: Document domain
        """
        if not self._enable_experience_bank or not self._experience_bank:
            return

        try:
            # Calculate quality score (simplified - use entity count as proxy)
            # In production, use actual quality evaluation
            quality_score = min(0.5 + (len(entities) * 0.05), 1.0)

            if quality_score >= 0.85:
                # Store experience for each chunk
                for chunk in chunks:
                    # Find entities for this chunk
                    chunk_entities = [
                        {
                            "name": e.name,
                            "entity_type": e.entity_type,
                            "description": e.description,
                            "confidence": e.confidence,
                        }
                        for e in entities
                        if hasattr(e, "chunk_id") and e.chunk_id == chunk.id
                    ]

                    chunk_edges = [
                        {
                            "source": str(e.source_id),
                            "target": str(e.target_id),
                            "relationship_type": e.edge_type.value
                            if hasattr(e.edge_type, "value")
                            else str(e.edge_type),
                        }
                        for e in edges
                    ]

                    if chunk_entities:
                        await self._experience_bank.store_experience(
                            text=chunk.text,
                            entities=chunk_entities,
                            relationships=chunk_edges,
                            quality_score=quality_score,
                            domain=domain,
                            chunk_id=chunk.id,
                            extraction_method="multi_agent",
                        )

                logger.info(
                    f"Stored {len(chunks)} extraction experiences (quality: {quality_score:.2f})"
                )

        except Exception as e:
            logger.warning(f"Failed to store extraction experience: {e}")

    async def evolve_prompts(
        self,
        domain: str,
        test_documents: list,
    ) -> dict:
        """Manually trigger prompt evolution for a domain.

        Args:
            domain: Domain to evolve prompts for
            test_documents: Test documents for evaluation

        Returns:
            Evolution statistics
        """
        if not self._enable_prompt_evolution or not self._prompt_evolution:
            return {"error": "Prompt evolution not enabled"}

        try:
            best_variant = await self._prompt_evolution.evolve_generation(
                domain=domain,
                test_documents=test_documents,
            )

            stats = self._prompt_evolution.get_domain_statistics(domain)

            return {
                "success": True,
                "domain": domain,
                "best_variant": {
                    "name": best_variant.name,
                    "quality_score": best_variant.avg_quality_score,
                },
                "statistics": stats,
            }

        except Exception as e:
            logger.error(f"Prompt evolution failed: {e}")
            return {"error": str(e)}

    async def get_experience_bank_stats(self) -> dict:
        """Get Experience Bank statistics.

        Returns:
            Statistics dictionary
        """
        if not self._enable_experience_bank or not self._experience_bank:
            return {"error": "Experience Bank not enabled"}

        try:
            return await self._experience_bank.get_experience_statistics()
        except Exception as e:
            return {"error": str(e)}
