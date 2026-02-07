from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from uuid import UUID
from .persistence.v1.schema import (
    Base,
    Entity,
    Edge,
    Document,
    ReviewQueue,
    ReviewStatus,
)
from datetime import datetime
import logging

from knowledge_base.intelligence import (
    HallucinationDetector,
    EntityVerification,
    RiskLevel,
)
from knowledge_base.common.resilient_gateway import ResilientGatewayClient

logger = logging.getLogger(__name__)


class ReviewService:
    """Service for managing human review queue for low-confidence entity resolutions and relationships."""

    def __init__(
        self, db_session: Session, enable_hallucination_detection: bool = True
    ):
        """
        Initialize the ReviewService.

        Args:
            db_session: SQLAlchemy database session
            enable_hallucination_detection: If True, use hallucination detection for prioritization
        """
        self.db = db_session
        self._enable_hallucination_detection = enable_hallucination_detection
        if enable_hallucination_detection:
            self._hallucination_detector = HallucinationDetector()
        else:
            self._hallucination_detector = None

    def add_entity_for_review(
        self,
        entity_id: UUID,
        confidence_score: float,
        grounding_quote: str = "",
        source_text: str = "",
        confidence_threshold: float = 0.7,
    ) -> Optional[ReviewQueue]:
        """
        Add an entity to the review queue if its confidence is below threshold.

        Args:
            entity_id: ID of the entity to potentially add for review
            confidence_score: The confidence score of the entity (0.0-1.0)
            grounding_quote: Quote that supports the resolution
            source_text: Original source text
            confidence_threshold: Confidence threshold (0.0-1.0) below which entities need review

        Returns:
            ReviewQueue object if entity was added for review, None otherwise
        """
        from sqlalchemy import select

        entity_result = self.db.execute(select(Entity).where(Entity.id == entity_id))
        entity = entity_result.scalar_one_or_none()
        if not entity:
            logger.warning(f"Entity with ID {entity_id} not found")
            return None

        if confidence_score >= confidence_threshold:
            logger.info(
                f"Entity {entity_id} has sufficient confidence ({confidence_score}) - no review needed"
            )
            return None

        # Check if already in review queue
        existing_review_result = self.db.execute(
            select(ReviewQueue).where(
                and_(
                    ReviewQueue.entity_id == entity_id, ReviewQueue.status == "pending"
                )
            )
        )
        existing_review = existing_review_result.scalar_one_or_none()

        if existing_review:
            logger.info(f"Entity {entity_id} already in review queue")
            return existing_review

        # Create new review entry
        review_entry = ReviewQueue(
            item_type="entity_resolution",
            entity_id=entity_id,
            edge_id=None,
            document_id=None,
            merged_entity_ids=[],  # Empty list for entity reviews
            confidence_score=confidence_score,
            grounding_quote=grounding_quote,
            source_text=source_text,
            status=ReviewStatus.PENDING,
            priority=self._calculate_priority(
                int(confidence_score * 100)
            ),  # Convert to 0-100 scale
            created_at=datetime.utcnow(),
        )

        self.db.add(review_entry)
        self.db.commit()
        self.db.refresh(review_entry)

        logger.info(
            f"Added entity {entity_id} to review queue with priority {review_entry.priority}"
        )
        return review_entry

    def add_edge_for_review(
        self, edge_id: UUID, confidence_score: float, confidence_threshold: float = 0.7
    ) -> Optional[ReviewQueue]:
        """
        Add an edge to the review queue if its confidence is below threshold.

        Args:
            edge_id: ID of the edge to potentially add for review
            confidence_score: The confidence score of the edge (0.0-1.0)
            confidence_threshold: Confidence threshold (0.0-1.0) below which edges need review

        Returns:
            ReviewQueue object if edge was added for review, None otherwise
        """
        from sqlalchemy import select

        edge_result = self.db.execute(select(Edge).where(Edge.id == edge_id))
        edge = edge_result.scalar_one_or_none()
        if not edge:
            logger.warning(f"Edge with ID {edge_id} not found")
            return None

        if confidence_score >= confidence_threshold:
            logger.info(
                f"Edge {edge_id} has sufficient confidence ({confidence_score}) - no review needed"
            )
            return None

        # Check if already in review queue
        existing_review_result = self.db.execute(
            select(ReviewQueue).where(
                and_(
                    ReviewQueue.edge_id == edge_id,
                    ReviewQueue.status == ReviewStatus.PENDING,
                )
            )
        )
        existing_review = existing_review_result.scalar_one_or_none()

        if existing_review:
            logger.info(f"Edge {edge_id} already in review queue")
            return existing_review

        # Create new review entry
        review_entry = ReviewQueue(
            item_type="edge_validation",
            entity_id=None,
            edge_id=edge_id,
            document_id=getattr(
                edge, "document_id", None
            ),  # Assuming edge has document_id
            merged_entity_ids=[],  # Empty for edge reviews
            confidence_score=confidence_score,
            grounding_quote="",  # Could be enhanced with actual data
            source_text="",  # Could be enhanced with actual source
            status=ReviewStatus.PENDING,
            priority=self._calculate_priority(
                int(confidence_score * 100)
            ),  # Convert to 0-100 scale
            created_at=datetime.utcnow(),
        )

        self.db.add(review_entry)
        self.db.commit()
        self.db.refresh(review_entry)

        logger.info(
            f"Added edge {edge_id} to review queue with priority {review_entry.priority}"
        )
        return review_entry

    def get_pending_reviews(
        self, limit: int = 50, offset: int = 0
    ) -> List[ReviewQueue]:
        """
        Get pending reviews ordered by priority (highest first).

        Args:
            limit: Maximum number of reviews to return
            offset: Number of reviews to skip

        Returns:
            List of ReviewQueue objects
        """
        from sqlalchemy import select

        result = self.db.execute(
            select(ReviewQueue)
            .where(ReviewQueue.status == "pending")
            .order_by(ReviewQueue.priority.desc(), ReviewQueue.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    def get_review_by_id(self, review_id: UUID) -> Optional[ReviewQueue]:
        """
        Get a specific review by ID.

        Args:
            review_id: ID of the review to retrieve

        Returns:
            ReviewQueue object or None if not found
        """
        from sqlalchemy import select

        result = self.db.execute(select(ReviewQueue).where(ReviewQueue.id == review_id))
        return result.scalar_one_or_none()

    def approve_review(
        self, review_id: UUID, reviewer_notes: Optional[str] = None
    ) -> bool:
        """
        Approve a review, marking it as completed.

        Args:
            review_id: ID of the review to approve
            reviewer_notes: Optional notes from the reviewer

        Returns:
            True if review was approved, False if review not found
        """
        from sqlalchemy import update

        # Use SQLAlchemy update statement for direct database update
        result = self.db.execute(
            update(ReviewQueue)
            .where(ReviewQueue.id == review_id)
            .values(
                status="approved",
                reviewer_notes=reviewer_notes,
                reviewed_at=datetime.utcnow(),
            )
        )

        self.db.commit()
        # Check if the review exists by re-querying it
        review_check = self.get_review_by_id(review_id)
        if not review_check:
            return False

        logger.info(f"Review {review_id} approved")
        return True

    def reject_review(
        self,
        review_id: UUID,
        corrections: Dict[str, Any],
        reviewer_notes: Optional[str] = None,
    ) -> bool:
        """
        Reject a review with corrections.

        Args:
            review_id: ID of the review to reject
            corrections: Dictionary of corrections to apply
            reviewer_notes: Optional notes from the reviewer

        Returns:
            True if review was rejected and corrections applied, False if review not found
        """
        from sqlalchemy import select, update

        # Get the review fresh from the database to ensure we have proper values
        review_result = self.db.execute(
            select(
                ReviewQueue.entity_id, ReviewQueue.edge_id, ReviewQueue.item_type
            ).where(ReviewQueue.id == review_id)
        )
        review_row = review_result.first()
        if not review_row:
            return False

        # Unpack the values to ensure they are Python values, not Column objects
        entity_id, edge_id, item_type = review_row

        # Apply corrections based on review type
        if entity_id is not None and item_type == "entity_resolution":
            entity_result = self.db.execute(
                select(Entity).where(Entity.id == entity_id)
            )
            entity = entity_result.scalar_one_or_none()
            if entity and corrections:
                update_stmt = update(Entity).where(Entity.id == entity_id)
                update_dict = {}
                for key, value in corrections.items():
                    if hasattr(Entity, key):
                        update_dict[key] = value
                if update_dict:
                    self.db.execute(update_stmt.values(**update_dict))
                    self.db.commit()
                logger.info(f"Applied corrections to entity {entity_id}: {corrections}")

        elif edge_id is not None and item_type == "edge_validation":
            edge_result = self.db.execute(select(Edge).where(Edge.id == edge_id))
            edge = edge_result.scalar_one_or_none()
            if edge and corrections:
                update_stmt = update(Edge).where(Edge.id == edge_id)
                update_dict = {}
                for key, value in corrections.items():
                    if hasattr(Edge, key):
                        update_dict[key] = value
                if update_dict:
                    self.db.execute(update_stmt.values(**update_dict))
                    self.db.commit()
                logger.info(f"Applied corrections to edge {edge_id}: {corrections}")

        # Update review status using direct update
        result = self.db.execute(
            update(ReviewQueue)
            .where(ReviewQueue.id == review_id)
            .values(
                status="rejected",
                reviewer_notes=reviewer_notes,
                reviewed_at=datetime.utcnow(),
            )
        )

        self.db.commit()
        # Check if the review exists by re-querying it
        review_check = self.get_review_by_id(review_id)
        if not review_check:
            return False

        logger.info(f"Review {review_id} rejected with corrections")
        return True

    def _calculate_priority(self, confidence: int) -> int:
        """
        Calculate review priority based on confidence score.
        Lower confidence = higher priority.

        Args:
            confidence: Confidence score (0-100)

        Returns:
            Priority level (1-10, 10 being highest priority)
        """
        # Map confidence to priority (lower confidence = higher priority)
        if confidence < 30:
            return 10
        elif confidence < 50:
            return 8
        elif confidence < 70:
            return 6
        else:
            return 4

    def _get_entity(self, entity_id: UUID) -> Optional[Entity]:
        """Get entity from database by ID."""
        from sqlalchemy import select

        entity_result = self.db.execute(select(Entity).where(Entity.id == entity_id))
        return entity_result.scalar_one_or_none()

    async def verify_entity_for_review(
        self, entity_id: UUID, source_text: str, confidence_score: float
    ) -> Tuple[bool, int, str]:
        """
        Verify entity using hallucination detection.

        Returns:
            Tuple of (should_auto_approve, priority, reason)
        """
        if self._hallucination_detector and self._enable_hallucination_detection:
            entity = self._get_entity(entity_id)
            if entity:
                try:
                    verification = await self._hallucination_detector.verify_entity(
                        entity=entity, context="", source_text=source_text
                    )

                    if (
                        confidence_score >= 0.85
                        and verification.risk_level == RiskLevel.LOW
                    ):
                        return True, 1, "auto_approved_high_confidence_low_risk"

                    priority = self._calculate_priority_with_hallucination(
                        confidence_score=confidence_score,
                        risk_level=verification.risk_level,
                    )
                    return False, priority, f"risk_{verification.risk_level.value}"
                except Exception as e:
                    logger.warning(
                        f"Hallucination detection failed for entity {entity_id}: {e}"
                    )
                    return (
                        False,
                        self._calculate_priority(int(confidence_score * 100)),
                        "confidence_based",
                    )

        return (
            False,
            self._calculate_priority(int(confidence_score * 100)),
            "confidence_based",
        )

    def _calculate_priority_with_hallucination(
        self, confidence_score: float, risk_level: RiskLevel
    ) -> int:
        """Calculate priority (1-10) based on confidence and hallucination risk."""
        risk_weights = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 1.0,
        }
        hallucination_factor = risk_weights.get(risk_level, 0.5)
        confidence_factor = 1 - confidence_score

        priority_score = (confidence_factor * 0.4) + (hallucination_factor * 0.6)
        return min(10, max(1, int(priority_score * 10) + 1))
