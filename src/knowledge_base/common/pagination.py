"""
AIP-158 compliant pagination utilities for standardized list endpoint responses.

This module provides pagination parameters and helper functions following Google's
API Improvement Proposal 158, ensuring consistent pagination across all list endpoints.
"""

from typing import TypeVar, List, Generic
from pydantic import BaseModel, Field, validator

T = TypeVar("T")


class PageParams(BaseModel):
    """Standard pagination parameters for list endpoints.

    Attributes:
        limit: Maximum number of items to return (default: 50, max: 1000)
        offset: Number of items to skip from the beginning (default: 0)
        order_by: Optional field to sort results (e.g., "name asc, created_at desc")
    """

    limit: int = Field(
        default=50, ge=1, le=1000, description="Maximum number of items to return"
    )
    offset: int = Field(
        default=0, ge=0, description="Number of items to skip from the beginning"
    )
    order_by: str | None = Field(
        default=None,
        description="Field to sort by with optional direction (e.g., 'name asc', 'created_at desc')",
    )

    @validator("limit")
    def limit_must_be_reasonable(cls, v: int) -> int:
        """Ensure limit is within reasonable bounds."""
        if v > 1000:
            raise ValueError("limit cannot exceed 1000 for performance reasons")
        return v

    def apply_to_query(self, query, model=None):
        """Apply pagination parameters to a SQLAlchemy query.

        Args:
            query: SQLAlchemy query object
            model: Optional SQLAlchemy model for sorting

        Returns:
            Modified query with pagination applied
        """
        from sqlalchemy import asc, desc

        # Apply ordering if specified
        if self.order_by and model:
            parts = self.order_by.split()
            if len(parts) == 1:
                field_name = parts[0]
                direction = "asc"
            else:
                field_name = parts[0]
                direction = parts[1].lower()

            if hasattr(model, field_name):
                field = getattr(model, field_name)
                if direction == "desc":
                    query = query.order_by(desc(field))
                else:
                    query = query.order_by(asc(field))

        # Apply pagination limits
        return query.limit(self.limit).offset(self.offset)


def create_paginated_response(
    items: List[T], total: int, limit: int, offset: int
) -> "PaginatedResponse[T]":
    """Helper function to create a paginated response.

    Args:
        items: List of items for the current page
        total: Total number of items across all pages
        limit: Maximum items per page
        offset: Number of items skipped

    Returns:
        PaginatedResponse with items and metadata
    """
    from knowledge_base.common.api_models import PaginatedResponse

    return PaginatedResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + len(items)) < total,
    )


def calculate_pagination_metadata(total: int, limit: int, offset: int) -> dict:
    """Calculate pagination metadata.

    Args:
        total: Total number of items
        limit: Maximum items per page
        offset: Number of items skipped

    Returns:
        Dictionary with pagination metadata
    """
    current_page = (offset // limit) + 1
    total_pages = (total + limit - 1) // limit if total > 0 and limit > 0 else 0

    return {
        "current_page": current_page,
        "total_pages": total_pages,
        "has_more": (offset + limit) < total,
        "has_previous": offset > 0,
    }
