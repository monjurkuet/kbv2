import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """FastAPI TestClient fixture for testing the API."""
    return TestClient(app)


@pytest.fixture
def sample_document():
    """Sample document fixture for testing."""
    return {
        "title": "Test Document",
        "content": "This is a test document for testing purposes.",
        "tags": ["test", "document", "pytest"],
    }


@pytest.fixture
def sample_user():
    """Sample user fixture for testing."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123",
    }


@pytest.fixture
def auth_headers():
    """Sample authentication headers fixture."""
    return {"Authorization": "Bearer test-token", "Content-Type": "application/json"}
