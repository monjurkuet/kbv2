"""Quality assurance service for validation and review management."""

import logging
from typing import List, Optional
from uuid import UUID

from knowledge_base.orchestration.base_service import BaseService
from knowledge_base.config.constants import MIN_EXTRACTION_QUALITY_SCORE
from knowledge_base.persistence.v1.schema import (
    Document,
    Entity,
    Edge,
    ReviewQueue,
    ReviewStatus,
)


class QualityAssuranceService(BaseService):
    """Service for quality validation and review management."""

    def __init__(self):
        super().__init__()
        self._domain_schema_validator = None
        self._hallucination_detector = None
        self._community_summarizer = None

    async def initialize(self) -> None:
        """Initialize the service."""
        self._logger.info("QualityAssuranceService initialized")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        self._logger.info("QualityAssuranceService shutdown")

    def set_validators(
        self,
        domain_schema_validator,
        hallucination_detector,
        community_summarizer,
    ):
        """Set the validation components."""
        self._domain_schema_validator = domain_schema_validator
        self._hallucination_detector = hallucination_detector
        self._community_summarizer = community_summarizer

    async def validate(
        self,
        document: Document,
        entities: List[Entity],
        edges: List[Edge],
        domain: str,
    ) -> dict:
        """Validate extracted entities and edges against domain schema.

        Args:
            document: The document
            entities: List of entities
            edges: List of edges
            domain: Document domain

        Returns:
            Validation results
        """
        if not self._domain_schema_validator:
            return {"valid": True, "issues": []}

        issues = []
        valid_entities = []

        for entity in entities:
            try:
                is_valid = await self._domain_schema_validator.validate_entity(
                    entity, domain
                )
                if is_valid:
                    valid_entities.append(entity)
                else:
                    issues.append(
                        {
                            "type": "invalid_entity",
                            "entity_id": entity.id,
                            "entity_name": entity.name,
                            "reason": "Entity type not in domain schema",
                        }
                    )
            except Exception as e:
                self._logger.error(f"Error validating entity {entity.name}: {e}")
                issues.append(
                    {
                        "type": "validation_error",
                        "entity_id": entity.id,
                        "error": str(e),
                    }
                )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "valid_entities": valid_entities,
        }

    async def detect_hallucinations(
        self,
        document: Document,
        entities: List[Entity],
        chunks: List,
    ) -> dict:
        """Detect hallucinated entities.

        Args:
            document: The document
            entities: List of entities
            chunks: List of chunks

        Returns:
            Hallucination detection results
        """
        if not self._hallucination_detector:
            return {"hallucinations": []}

        hallucinations = []

        for entity in entities:
            try:
                verification = await self._hallucination_detector.verify_entity(
                    entity, document, chunks
                )

                if not verification.is_supported:
                    hallucinations.append(
                        {
                            "entity_id": entity.id,
                            "entity_name": entity.name,
                            "confidence": verification.confidence,
                            "reason": verification.reason,
                        }
                    )
            except Exception as e:
                self._logger.error(
                    f"Error detecting hallucination for {entity.name}: {e}"
                )

        return {
            "hallucinations": hallucinations,
            "total_checked": len(entities),
            "hallucination_count": len(hallucinations),
        }

    async def generate_reports(
        self,
        document: Document,
        entities: List[Entity],
        edges: List[Edge],
    ) -> None:
        """Generate community summaries and reports.

        Args:
            document: The document
            entities: List of entities
            edges: List of edges
        """
        if not self._community_summarizer:
            self._logger.warning(
                "Community summarizer not set, skipping report generation"
            )
            return

        try:
            summary = await self._community_summarizer.generate_multi_level_summary(
                entities=entities,
                edges=edges,
                document_id=document.id,
            )

            self._logger.info(
                f"Generated community summaries for document '{document.name}'"
            )

        except Exception as e:
            self._logger.error(f"Error generating reports: {e}")

    async def route_to_review(
        self,
        items: List[dict],
        session,
    ) -> None:
        """Route items to the review queue.

        Args:
            items: List of items to review (each with item_type, item_id, reason)
            session: Database session
        """
        review_items = []

        for item in items:
            review_item = ReviewQueue(
                item_type=item["item_type"],
                item_id=item["item_id"],
                reason=item.get("reason", "Manual review required"),
                priority=item.get("priority", 5),
                metadata=item.get("metadata", {}),
            )
            review_items.append(review_item)

        session.add_all(review_items)
        await session.commit()

        self._logger.info(f"Routed {len(review_items)} items to review queue")
