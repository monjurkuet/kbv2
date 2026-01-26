from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any, Generator
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID

from .persistence.v1.schema import (
    Base,
    ReviewQueue,
    Entity,
    Edge,
    Document,
    ReviewStatus,
)
from .review_service import ReviewService


# Create a database dependency function
def get_db() -> Generator[Session, None, None]:
    """Dependency to provide database session.

    Yields:
        Session: SQLAlchemy database session.
    """
    # This is a placeholder - in a real implementation, this would return the actual session
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import create_engine
    import os

    # Use database URL from environment or default
    database_url = os.getenv(
        "DATABASE_URL", "postgresql://agentzero@localhost:5432/knowledge_base"
    )
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


router = APIRouter(prefix="/api/v1/review", tags=["review"])


class ReviewItem(BaseModel):
    """Pydantic model for review queue items."""

    id: UUID
    item_type: str
    entity_id: Optional[UUID] = None
    edge_id: Optional[UUID] = None
    document_id: Optional[UUID] = None
    merged_entity_ids: Optional[List[UUID]] = None
    confidence_score: Optional[float] = None
    grounding_quote: Optional[str] = None
    source_text: Optional[str] = None
    status: str
    priority: int
    created_at: datetime
    reviewed_at: Optional[datetime] = None
    reviewer_notes: Optional[str] = None

    class Config:
        from_attributes = True


class ReviewApproval(BaseModel):
    """Pydantic model for approving a review."""

    reviewer_notes: Optional[str] = None


class ReviewRejection(BaseModel):
    """Pydantic model for rejecting a review with corrections."""

    corrections: Dict[str, Any]
    reviewer_notes: Optional[str] = None


class QualityMetricsResponse(BaseModel):
    """Response for quality metrics endpoint."""

    total_pending: int
    by_priority: Dict[str, int]
    hallucination_distribution: Dict[str, int]
    auto_approved_count: int
    recommendation: str
    generated_at: datetime


class DomainQualityResponse(BaseModel):
    """Response for domain-specific quality metrics."""

    domain: str
    total_entities: int
    avg_quality_score: float
    hallucination_rate: float
    auto_approval_rate: float


@router.get("/pending", response_model=List[ReviewItem])
async def get_pending_reviews(
    limit: int = 50, offset: int = 0, db: Session = Depends(get_db)
):
    """
    Get pending reviews ordered by priority.

    Args:
        limit: Maximum number of reviews to return (default: 50)
        offset: Number of reviews to skip (default: 0)
        db: Database session dependency

    Returns:
        List of pending review items
    """
    review_service = ReviewService(db)
    reviews = review_service.get_pending_reviews(limit=limit, offset=offset)
    return reviews


@router.get("/{review_id}", response_model=ReviewItem)
async def get_review(review_id: UUID, db: Session = Depends(get_db)):
    """
    Get a specific review by ID.

    Args:
        review_id: ID of the review to retrieve
        db: Database session dependency

    Returns:
        Review item

    Raises:
        HTTPException: If review not found
    """
    review_service = ReviewService(db)
    review = review_service.get_review_by_id(review_id)
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review with ID {review_id} not found",
        )
    return review


@router.post("/{review_id}/approve", response_model=bool)
async def approve_review(
    review_id: UUID, approval: ReviewApproval, db: Session = Depends(get_db)
):
    """
    Approve a review.

    Args:
        review_id: ID of the review to approve
        approval: Approval data including optional reviewer notes
        db: Database session dependency

    Returns:
        True if review was approved

    Raises:
        HTTPException: If review not found
    """
    review_service = ReviewService(db)
    success = review_service.approve_review(review_id, approval.reviewer_notes)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review with ID {review_id} not found",
        )
    return success


@router.post("/{review_id}/reject", response_model=bool)
async def reject_review(
    review_id: UUID, rejection: ReviewRejection, db: Session = Depends(get_db)
):
    """
    Reject a review with corrections.

    Args:
        review_id: ID of the review to reject
        rejection: Rejection data including corrections and optional reviewer notes
        db: Database session dependency

    Returns:
        True if review was rejected and corrections applied

    Raises:
        HTTPException: If review not found
    """
    review_service = ReviewService(db)
    success = review_service.reject_review(
        review_id, rejection.corrections, rejection.reviewer_notes
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review with ID {review_id} not found",
        )
    return success


@router.get("/quality-metrics", response_model=QualityMetricsResponse)
async def get_quality_metrics(
    db: Session = Depends(get_db), domain: Optional[str] = None
) -> QualityMetricsResponse:
    """Get extraction quality metrics overview.

    Args:
        db: Database session
        domain: Optional domain filter

    Returns:
        Quality metrics including hallucination distribution and priority breakdown
    """
    service = ReviewService(db)

    reviews = service.get_pending_reviews(limit=1000)

    total = len(reviews)

    by_priority = {}
    auto_approved = 0
    for review in reviews:
        priority = review.priority
        by_priority[str(priority)] = by_priority.get(str(priority), 0) + 1
        if priority <= 2:
            auto_approved += 1

    hallucination_distribution = {
        "critical": by_priority.get("9", 0) + by_priority.get("10", 0),
        "high": by_priority.get("7", 0) + by_priority.get("8", 0),
        "medium": by_priority.get("4", 0)
        + by_priority.get("5", 0)
        + by_priority.get("6", 0),
        "low": by_priority.get("1", 0)
        + by_priority.get("2", 0)
        + by_priority.get("3", 0),
    }

    critical_count = hallucination_distribution["critical"]
    if critical_count > 10:
        recommendation = "URGENT: Prioritize reviews with priority >= 9 immediately"
    elif critical_count > 0:
        recommendation = "Review priority 9-10 items in the next batch"
    elif hallucination_distribution["high"] > 5:
        recommendation = "Focus on high-priority items (priority 7-8)"
    else:
        recommendation = "Queue is healthy - process in priority order"

    return QualityMetricsResponse(
        total_pending=total,
        by_priority=by_priority,
        hallucination_distribution=hallucination_distribution,
        auto_approved_count=auto_approved,
        recommendation=recommendation,
        generated_at=datetime.utcnow(),
    )


@router.get("/domain-quality/{domain}", response_model=DomainQualityResponse)
async def get_domain_quality(
    domain: str, db: Session = Depends(get_db)
) -> DomainQualityResponse:
    """Get quality metrics for a specific domain."""
    from sqlalchemy import func

    service = ReviewService(db)
    reviews = service.get_pending_reviews(limit=1000)

    domain_reviews = [r for r in reviews if hasattr(r, "domain") and r.domain == domain]
    total = len(domain_reviews)

    if total == 0:
        return DomainQualityResponse(
            domain=domain,
            total_entities=0,
            avg_quality_score=0.0,
            hallucination_rate=0.0,
            auto_approval_rate=0.0,
        )

    auto_approved = sum(1 for r in reviews if r.priority <= 2)

    hallucination_rate = sum(1 for r in domain_reviews if r.priority >= 7) / max(
        total, 1
    )
    auto_approval_rate = auto_approved / max(len(reviews), 1)

    avg_quality = sum(r.confidence_score or 0.5 for r in domain_reviews) / max(total, 1)

    return DomainQualityResponse(
        domain=domain,
        total_entities=total,
        avg_quality_score=avg_quality,
        hallucination_rate=hallucination_rate,
        auto_approval_rate=auto_approval_rate,
    )
