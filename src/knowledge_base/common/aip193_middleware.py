"""
AIP-193 Response Middleware for automatic response wrapping.

This middleware automatically wraps all API responses in the AIP-193 compliant
format, ensuring consistent response structure across all endpoints without
requiring manual wrapping in each endpoint.
"""

import json
import logging
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message

from knowledge_base.common.api_models import APIError, ErrorInfo

logger = logging.getLogger(__name__)


class AIP193ResponseMiddleware(BaseHTTPMiddleware):
    """
    Automatically wraps all endpoint responses in AIP-193 structure.

    This middleware intercepts all responses and wraps them in the standard
    APIResponse format. It handles both successful responses and errors,
    ensuring 100% AIP-193 compliance across the entire API surface.

    Benefits:
    - No manual response wrapping needed in endpoints
    - Consistent response format for all APIs (query, review, graph, document)
    - Automatic error formatting with proper structure
    - Maintains backward compatibility for non-JSON responses
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """
        Process incoming request and wrap outgoing response.

        Args:
            request: Incoming FastAPI request
            call_next: Next middleware or endpoint in chain

        Returns:
            Response wrapped in AIP-193 format
        """
        # Skip health check and non-API endpoints
        path = request.url.path
        if path in ["/health", "/ready", "/metrics"] or not path.startswith("/api"):
            return await call_next(request)

        # Process the request and get response
        response = await call_next(request)

        # Skip if response is not JSON or empty
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return response

        # Read response body
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        # If response is already wrapped (e.g., from error handler), return as-is
        try:
            body_text = response_body.decode("utf-8")
            if not body_text or not body_text.strip():
                # Empty response, wrap as success with null data
                return self._create_wrapped_response(
                    {"success": True, "data": None, "error": None, "metadata": {}}
                )

            body_data = json.loads(body_text)

            # Check if already wrapped (has success field) or is error response
            if isinstance(body_data, dict) and (
                "success" in body_data or "error" in body_data
            ):
                # Already wrapped, return as-is
                return Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )

            # Wrap successful response
            wrapped_response = {
                "success": response.status_code < 400,
                "data": body_data,
                "error": None,
                "metadata": {
                    "request_id": request.headers.get("x-request-id"),
                    "path": path,
                    "method": request.method,
                    "timestamp": request.headers.get("x-request-timestamp"),
                },
            }

            return self._create_wrapped_response(wrapped_response, response.status_code)

        except json.JSONDecodeError:
            # Body is not valid JSON, return original response
            logger.warning(f"Non-JSON response from {path}, skipping AIP-193 wrapper")
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
            )
        except Exception as e:
            # Unexpected error during wrapping
            logger.error(f"Error wrapping response for {path}: {e}", exc_info=True)

            # Return error response in AIP-193 format
            error_response = self._create_error_response(
                status_code=500,
                message="Internal server error during response processing",
                error_code="RESPONSE_WRAPPER_ERROR",
                details={"original_path": path, "error": str(e)},
            )

            return self._create_wrapped_response(error_response)

    def _create_wrapped_response(
        self, data: dict[str, Any], original_status_code: int = 200
    ) -> Response:
        """
        Create a wrapped JSON response.

        Args:
            data: Response dictionary
            original_status_code: Original HTTP status code

        Returns:
            JSON Response with correct headers
        """
        return Response(
            content=json.dumps(data, default=str),
            status_code=200,  # AIP-193: Always 200 for JSON responses
            headers={
                "content-type": "application/json",
                "x-original-status": str(
                    original_status_code
                ),  # Preserve original code
            },
        )

    def _create_error_response(
        self,
        status_code: int,
        message: str,
        error_code: str | None = None,
        details: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Create an AIP-193 error response.

        Args:
            status_code: HTTP status code
            message: Error message
            error_code: Optional custom error code
            details: Optional additional context

        Returns:
            Error response dictionary
        """
        status_map = {
            400: "INVALID_ARGUMENT",
            401: "UNAUTHENTICATED",
            403: "PERMISSION_DENIED",
            404: "NOT_FOUND",
            409: "ALREADY_EXISTS",
            500: "INTERNAL",
            503: "UNAVAILABLE",
        }

        status_str = status_map.get(status_code, "UNKNOWN")
        if error_code is None:
            error_code = status_str

        error_info = ErrorInfo(
            reason=error_code,
            metadata=details or {},
        )

        error = APIError(
            code=status_code, message=message, status=status_str, details=[error_info]
        )

        return {
            "success": False,
            "error": error.dict(),
            "data": None,
            "metadata": details or {},
        }
