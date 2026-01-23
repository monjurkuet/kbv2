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
