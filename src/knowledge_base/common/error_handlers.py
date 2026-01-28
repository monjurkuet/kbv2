"""
Global exception handlers for AIP-193 compliant error responses.

This module provides centralized error handling that converts all exceptions
into standardized API error responses following Google's API Improvement
Proposal 193.
"""

from typing import Dict, Any
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from knowledge_base.common.api_models import APIError, ErrorInfo
import logging

logger = logging.getLogger(__name__)


def create_api_error(
    status_code: int,
    message: str,
    error_code: str | None = None,
    details: Dict[str, str] | None = None,
) -> APIError:
    """
    Create an AIP-193 compliant error object.

    Args:
        status_code: HTTP status code
        message: Human-readable error message
        error_code: Optional canonical error code (e.g., "RESOURCE_NOT_FOUND")
        details: Optional additional context as key-value pairs

    Returns:
        APIError object with proper AIP-193 structure
    """
    # Map HTTP status codes to standard error status strings
    status_map = {
        400: "INVALID_ARGUMENT",
        401: "UNAUTHENTICATED",
        403: "PERMISSION_DENIED",
        404: "NOT_FOUND",
        409: "ALREADY_EXISTS",
        412: "FAILED_PRECONDITION",
        413: "REQUEST_ENTITY_TOO_LARGE",
        429: "RESOURCE_EXHAUSTED",
        499: "CANCELLED",
        500: "INTERNAL",
        501: "NOT_IMPLEMENTED",
        503: "UNAVAILABLE",
        504: "DEADLINE_EXCEEDED",
    }

    status_str = status_map.get(status_code, "UNKNOWN")
    if error_code is None:
        error_code = status_str

    error_info = ErrorInfo(reason=error_code, metadata=details or {})

    return APIError(
        code=status_code, message=message, status=status_str, details=[error_info]
    )


async def validation_exception_handler(
    request: Request, exc: ValidationError
) -> JSONResponse:
    """
    Pydantic validation error handler.

    Converts Pydantic validation errors into AIP-193 compliant responses
    with detailed field-level error information.

    Args:
        request: FastAPI request object
        exc: Pydantic ValidationError

    Returns:
        JSONResponse with AIP-193 error structure
    """
    # Extract field validation errors
    error_details = {}
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        error_message = error["msg"]
        error_details[field] = error_message

    method = getattr(request, "method", "WS")
    path = getattr(request.url, "path", "unknown") if hasattr(request, "url") else "unknown"

    logger.warning(
        f"Validation error for {method} {path}",
        extra={
            "method": method,
            "path": path,
            "errors": error_details,
        },
    )

    error = create_api_error(
        status_code=status.HTTP_400_BAD_REQUEST,
        message="Invalid request parameters",
        error_code="VALIDATION_ERROR",
        details=error_details,
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,  # AIP-193: Always 200 for JSON errors
        content={
            "success": False,
            "error": error.dict(),
            "data": None,
            "metadata": {
                "request_id": getattr(request, "headers", {}).get("x-request-id") if hasattr(request, "headers") else None,
                "path": path,
            },
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    FastAPI HTTP exception handler.

    Converts FastAPI HTTPExceptions into AIP-193 compliant responses.

    Args:
        request: FastAPI request object
        exc: FastAPI HTTPException

    Returns:
        JSONResponse with AIP-193 error structure
    """
    path = getattr(request.url, "path", "unknown") if hasattr(request, "url") else "unknown"

    logger.info(
        f"HTTP exception: {exc.status_code} - {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "detail": str(exc.detail),
            "path": path,
        },
    )

    error = create_api_error(
        status_code=exc.status_code,
        message=str(exc.detail),
        details={"path": path},
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,  # AIP-193: Always 200 for JSON errors
        content={
            "success": False,
            "error": error.dict(),
            "data": None,
            "metadata": {
                "request_id": getattr(request, "headers", {}).get("x-request-id") if hasattr(request, "headers") else None,
                "path": path,
            },
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all exception handler for unexpected errors.

    Handles any unhandled exceptions and converts them into AIP-193 compliant
    error responses without leaking internal implementation details.

    Args:
        request: FastAPI request object
        exc: Any exception

    Returns:
        JSONResponse with AIP-193 error structure
    """
    error_message = "Internal server error"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    # Handle specific known exceptions
    if isinstance(exc, ValueError) and str(exc):
        error_message = str(exc)
        status_code = status.HTTP_400_BAD_REQUEST

    method = getattr(request, "method", "WS")
    path = getattr(request.url, "path", "unknown") if hasattr(request, "url") else "unknown"

    logger.error(
        f"Unexpected error processing {method} {path}",
        exc_info=True,
        extra={
            "method": method,
            "path": path,
            "error_type": exc.__class__.__name__,
        },
    )

    error = create_api_error(
        status_code=status_code,
        message=error_message,
        error_code="INTERNAL_ERROR",
        details={
            "type": exc.__class__.__name__,
            "request_id": request.headers.get("x-request-id", "unknown"),
        },
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,  # AIP-193: Always 200 for JSON errors
        content={
            "success": False,
            "error": error.dict(),
            "data": None,
            "metadata": {
                "request_id": getattr(request, "headers", {}).get("x-request-id") if hasattr(request, "headers") else "unknown",
                "path": path,
                "support_id": getattr(request, "headers", {}).get("x-request-id") if hasattr(request, "headers") else "unknown",
            },
        },
    )


async def resource_not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handler for ResourceNotFoundError (placeholder for custom exception class).

    This is a template for handling domain-specific exceptions in an AIP-193
    compliant manner.

    Args:
        request: FastAPI request object
        exc: ResourceNotFoundError (to be implemented)

    Returns:
        JSONResponse with AIP-193 error structure
    """
    error = create_api_error(
        status_code=status.HTTP_404_NOT_FOUND,
        message="Resource not found",
        error_code="RESOURCE_NOT_FOUND",
        details={
            "path": request.url.path,
            "resource_id": getattr(exc, "resource_id", "unknown"),
        },
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,  # AIP-193: Always 200 for JSON errors
        content={
            "success": False,
            "error": error.dict(),
            "data": None,
            "metadata": {
                "request_id": request.headers.get("x-request-id"),
                "path": request.url.path,
            },
        },
    )


def setup_exception_handlers(app):
    """
    Register all exception handlers with a FastAPI application.

    Args:
        app: FastAPI application instance
    """
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    # Add custom exception handlers here as needed
    app.add_exception_handler(Exception, general_exception_handler)
