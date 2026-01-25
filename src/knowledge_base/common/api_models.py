"""
AIP-193 compliant response models and error handling structures.

This module provides standardized API response wrappers following Google's
API Improvement Proposal 193, ensuring consistent response formats across
all endpoints.
"""

from typing import Generic, TypeVar, Optional, Dict, Any, List
from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorInfo(BaseModel):
    """Detailed error information following google.rpc.ErrorInfo pattern."""

    reason: str = Field(..., description="Error reason code")
    domain: str = Field(default="kb.v2.api", description="Error domain identifier")
    metadata: Dict[str, str] = Field(
        default_factory=dict, description="Additional error context"
    )


class APIError(BaseModel):
    """AIP-193 compliant error structure."""

    code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Human-readable error message")
    status: str = Field(..., description="google.rpc.Code enum value (e.g., NOT_FOUND)")
    details: List[ErrorInfo] = Field(
        default_factory=list, description="Detailed error information"
    )


class APIResponse(BaseModel, Generic[T]):
    """Standard wrapper for all API responses (AIP-193 compliance).

    Every API endpoint must return data wrapped in this structure to ensure
    consistent response parsing across the entire API surface.

    Attributes:
        success: Indicates whether the request was successful
        data: Response payload for successful requests
        error: Error details for failed requests
        metadata: Additional metadata (pagination, timing, request_id, etc.)
    """

    success: bool = Field(..., description="Indicates if request succeeded")
    data: Optional[T] = Field(None, description="Response payload on success")
    error: Optional[APIError] = Field(None, description="Error details on failure")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata (pagination, timing, request_id)",
    )


class PaginatedResponse(BaseModel, Generic[T]):
    """AIP-158 compliant pagination wrapper.

    Used for list endpoints that return multiple items with pagination support.
    """

    items: List[T] = Field(..., description="List of items in current page")
    total: int = Field(..., description="Total number of items across all pages", ge=0)
    limit: int = Field(..., description="Maximum items per page", ge=1, le=1000)
    offset: int = Field(..., description="Number of items skipped", ge=0)
    has_more: bool = Field(
        ..., description="Whether more items exist beyond current page"
    )
