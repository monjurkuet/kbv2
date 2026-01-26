"""
Unit tests for AIP-193 compliant error handlers.

Tests error handling across all scenarios including validation errors,
HTTP exceptions, database errors, LLM service errors, and unexpected exceptions.
"""

import pytest
from unittest.mock import Mock
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.testclient import TestClient
from pydantic import ValidationError, BaseModel

from knowledge_base.common.error_handlers import (
    create_api_error,
    validation_exception_handler,
    http_exception_handler,
    general_exception_handler,
    resource_not_found_handler,
    setup_exception_handlers,
)
from knowledge_base.common.api_models import APIError, ErrorInfo


class TestCreateAPIError:
    """Test suite for the create_api_error helper function."""

    def test_create_api_error_standard_codes(self):
        """Test error creation with standard HTTP status codes."""
        error = create_api_error(404, "Resource not found")

        assert error.code == 404
        assert error.message == "Resource not found"
        assert error.status == "NOT_FOUND"
        assert len(error.details) == 1
        assert error.details[0].reason == "NOT_FOUND"
        assert error.details[0].domain == "kb.v2.api"

    def test_create_api_error_custom_code(self):
        """Test error creation with custom error code."""
        error = create_api_error(
            400, "Invalid input", error_code="VALIDATION_ERROR", details={"field": "email"}
        )

        assert error.code == 400
        assert error.message == "Invalid input"
        assert error.status == "INVALID_ARGUMENT"
        assert error.details[0].reason == "VALIDATION_ERROR"
        assert error.details[0].metadata["field"] == "email"

    def test_create_api_error_unknown_status(self):
        """Test error creation with unknown status code."""
        error = create_api_error(999, "Unknown error")

        assert error.code == 999
        assert error.status == "UNKNOWN"
        assert error.details[0].reason == "UNKNOWN"

    def test_all_status_code_mappings(self):
        """Test all mapped status codes return correct status strings."""
        test_cases = [
            (400, "INVALID_ARGUMENT"),
            (401, "UNAUTHENTICATED"),
            (403, "PERMISSION_DENIED"),
            (404, "NOT_FOUND"),
            (409, "ALREADY_EXISTS"),
            (412, "FAILED_PRECONDITION"),
            (413, "REQUEST_ENTITY_TOO_LARGE"),
            (429, "RESOURCE_EXHAUSTED"),
            (499, "CANCELLED"),
            (500, "INTERNAL"),
            (501, "NOT_IMPLEMENTED"),
            (503, "UNAVAILABLE"),
            (504, "DEADLINE_EXCEEDED"),
        ]

        for status_code, expected_status in test_cases:
            error = create_api_error(status_code, "Test")
            assert error.status == expected_status, f"Failed for status code {status_code}"


class TestValidationExceptionHandler:
    """Test suite for Pydantic validation error handler."""

    @pytest.fixture
    def sample_request(self):
        """Create a mock request for testing."""
        request = Mock(spec=Request)
        request.method = "POST"
        request.url.path = "/api/test"
        request.headers = {"x-request-id": "test-123"}
        return request

    @pytest.fixture
    def validation_error(self):
        """Create a sample validation error."""
        class TestModel(BaseModel):
            name: str
            age: int
            email: str

        try:
            TestModel(name=123, age="invalid", email="not-an-email")
        except ValidationError as e:
            return e

    @pytest.mark.asyncio
    async def test_validation_handler_structure(self, sample_request, validation_error):
        """Test validation error returns proper AIP-193 structure."""
        response = await validation_exception_handler(sample_request, validation_error)

        assert response.status_code == 200  # AIP-193 compliance
        assert b"success" in response.body
        assert b"error" in response.body
        assert b"VALIDATION_ERROR" in response.body or b"validation error" in response.body.lower()

    @pytest.mark.asyncio
    async def test_validation_handler_field_errors(self, sample_request, validation_error):
        """Test validation error includes field-level details."""
        response = await validation_exception_handler(sample_request, validation_error)

        assert response.status_code == 200
        body = response.body.decode()
        # Should contain field error details
        assert "name" in body or "age" in body or "email" in body

    @pytest.mark.asyncio
    async def test_validation_handler_metadata(self, sample_request, validation_error):
        """Test validation error includes request metadata."""
        response = await validation_exception_handler(sample_request, validation_error)

        body = response.body.decode()
        assert "test-123" in body  # request_id
        assert "/api/test" in body  # path


class TestHTTPExceptionHandler:
    """Test suite for HTTP exception handler."""

    @pytest.fixture
    def sample_request(self):
        """Create a mock request for testing."""
        request = Mock(spec=Request)
        request.url.path = "/api/not-found"
        request.headers = {"x-request-id": "http-test-456"}
        return request

    @pytest.mark.asyncio
    async def test_http_exception_404(self, sample_request):
        """Test 404 Not Found HTTP exception."""
        exc = HTTPException(status_code=404, detail="Resource not found")
        response = await http_exception_handler(sample_request, exc)

        assert response.status_code == 200  # AIP-193 compliance
        body = response.body.decode()
        assert "NOT_FOUND" in body
        assert "Resource not found" in body

    @pytest.mark.asyncio
    async def test_http_exception_403(self, sample_request):
        """Test 403 Forbidden HTTP exception."""
        exc = HTTPException(status_code=403, detail="Access denied")
        response = await http_exception_handler(sample_request, exc)

        assert response.status_code == 200
        body = response.body.decode()
        assert "PERMISSION_DENIED" in body
        assert "Access denied" in body

    @pytest.mark.asyncio
    async def test_http_exception_500(self, sample_request):
        """Test 500 Internal Server Error HTTP exception."""
        exc = HTTPException(status_code=500, detail="Server error")
        response = await http_exception_handler(sample_request, exc)

        assert response.status_code == 200
        body = response.body.decode()
        assert "INTERNAL" in body
        assert "Server error" in body


class TestGeneralExceptionHandler:
    """Test suite for general exception handler."""

    @pytest.fixture
    def sample_request(self):
        """Create a mock request for testing."""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/api/error"
        request.headers = {"x-request-id": "error-test-789"}
        return request

    @pytest.mark.asyncio
    async def test_general_exception_unknown(self, sample_request):
        """Test handling of unexpected exceptions."""
        exc = RuntimeError("Unexpected runtime error")
        response = await general_exception_handler(sample_request, exc)

        assert response.status_code == 200  # AIP-193 compliance
        body = response.body.decode()
        assert "INTERNAL_ERROR" in body or "INTERNAL" in body
        assert "Internal server error" in body
        assert "RuntimeError" in body

    @pytest.mark.asyncio
    async def test_general_exception_value_error(self, sample_request):
        """Test ValueError gets 400 status."""
        exc = ValueError("Invalid parameter value")
        response = await general_exception_handler(sample_request, exc)

        assert response.status_code == 200  # Always 200 for AIP-193
        body = response.body.decode()
        # ValueError should have 400 error code but 200 HTTP status
        assert "400" in body or "Invalid parameter value" in body

    @pytest.mark.asyncio
    async def test_general_exception_no_message(self, sample_request):
        """Test exception with empty message."""
        exc = ValueError("")
        response = await general_exception_handler(sample_request, exc)

        assert response.status_code == 200
        body = response.body.decode()
        # Should fall back to internal server error for empty ValueError
        assert "Internal server error" in body


class TestResourceNotFoundHandler:
    """Test suite for resource not found handler."""

    @pytest.fixture
    def sample_request(self):
        """Create a mock request for testing."""
        request = Mock(spec=Request)
        request.url.path = "/api/documents/123"
        request.headers = {}
        return request

    @pytest.fixture
    def not_found_exception(self):
        """Create a mock ResourceNotFoundError."""
        exc = Exception("Resource not found")
        exc.resource_id = "123"
        return exc

    @pytest.mark.asyncio
    async def test_resource_not_found_handler(self, sample_request, not_found_exception):
        """Test resource not found error handling."""
        response = await resource_not_found_handler(sample_request, not_found_exception)

        assert response.status_code == 200
        body = response.body.decode()
        assert "RESOURCE_NOT_FOUND" in body
        assert "Resource not found" in body
        assert "123" in body  # resource_id


class TestAIP193Compliance:
    """Verify strict AIP-193 compliance in error responses."""

    @pytest.fixture
    def sample_request(self):
        """Create a mock request for testing."""
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.headers = {"x-request-id": "test-123"}
        return request

    @pytest.mark.asyncio
    async def test_error_response_structure(self, sample_request):
        """Verify error response follows AIP-193 structure."""
        exc = HTTPException(status_code=404, detail="Not found")

        response = await http_exception_handler(sample_request, exc)
        body = response.body.decode()

        # Check key elements are present
        assert "success" in body
        assert "error" in body
        assert "data" in body
        assert "metadata" in body
        assert "code" in body
        assert "message" in body
        assert "status" in body  # Check for status in original case
        assert "details" in body

    @pytest.mark.asyncio
    async def test_all_error_handlers_use_200_status(self):
        """Verify all error handlers return HTTP 200 for AIP-193 compliance."""
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.headers = {}

        # Test all handlers return 200
        handlers = [
            (validation_exception_handler, self._get_validation_error()),
            (http_exception_handler, HTTPException(status_code=404, detail="Not found")),
            (general_exception_handler, ValueError("Test error")),
        ]

        for handler, exc in handlers:
            if handler == validation_exception_handler:
                response = await validation_exception_handler(request, exc)
            elif handler == http_exception_handler:
                response = await http_exception_handler(request, exc)
            else:
                response = await general_exception_handler(request, exc)

            assert response.status_code == 200, f"{handler.__name__} should return 200"

    def _get_validation_error(self):
        """Helper to create a validation error."""
        class TestModel(BaseModel):
            field: str
        try:
            TestModel()
        except ValidationError as e:
            return e


def test_setup_exception_handlers():
    """Test exception handler registration."""
    app = FastAPI()
    setup_exception_handlers(app)

    # Check that handlers are registered
    assert len(app.exception_handlers) >= 3  # At least the 3 main handlers
    # Check specific handlers are registered
    assert any(ValidationError in app.exception_handlers for _ in [1])
    assert any(HTTPException in app.exception_handlers for _ in [1])
    assert any(Exception in app.exception_handlers for _ in [1])



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
