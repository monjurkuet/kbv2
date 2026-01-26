"""Unit tests for the Review API endpoints.

Tests the review queue management including pending review retrieval,
approval, rejection, and reprocessing functionality.
"""

from unittest.mock import Mock, patch
import pytest
from fastapi.testclient import TestClient
from uuid import uuid4, UUID
from knowledge_base.main import app

client = TestClient(app)


def test_get_pending_reviews_success():
    """Test retrieving pending reviews from the queue."""
    mock_reviews = [
        {
            "id": uuid4(),
            "item_type": "entity",
            "entity_id": uuid4(),
            "confidence_score": 0.85,
            "grounding_quote": "The AI system detected in paragraph 3",
            "source_text": "Artificial intelligence was used in the analysis",
            "status": "pending",
            "priority": 5,
            "created_at": "2024-01-20T10:00:00",
            "reviewed_at": None,
            "reviewer_notes": None
        },
        {
            "id": uuid4(),
            "item_type": "edge",
            "edge_id": uuid4(),
            "confidence_score": 0.72,
            "grounding_quote": "Relationship mentioned in section 2.1",
            "source_text": "Machine learning is a subset of AI",
            "status": "pending",
            "priority": 3,
            "created_at": "2024-01-19T15:30:00",
            "reviewed_at": None,
            "reviewer_notes": None
        }
    ]

    with patch('knowledge_base.review_api.ReviewService') as mock_service_class:
        mock_service = Mock()
        mock_service.get_pending_reviews.return_value = mock_reviews
        mock_service_class.return_value = mock_service

        response = client.get("/api/v1/review/pending")

        assert response.status_code == 200
        result = response.json()
        assert len(result["data"]) == 2
        assert result["data"][0]["item_type"] == "entity"
        assert result["data"][0]["status"] == "pending"
        assert result["data"][1]["item_type"] == "edge"
        assert "confidence_score" in result["data"][0]
        mock_service.get_pending_reviews.assert_called_once_with(limit=50, offset=0)


def test_get_pending_reviews_with_pagination():
    """Test retrieving pending reviews with pagination parameters."""
    with patch('knowledge_base.review_api.ReviewService') as mock_service_class:
        mock_service = Mock()
        mock_service.get_pending_reviews.return_value = []
        mock_service_class.return_value = mock_service

        response = client.get("/api/v1/review/pending?limit=10&offset=20")

        assert response.status_code == 200
        mock_service.get_pending_reviews.assert_called_once_with(limit=10, offset=20)


def test_get_pending_reviews_empty():
    """Test retrieving pending reviews when queue is empty."""
    with patch('knowledge_base.review_api.ReviewService') as mock_service_class:
        mock_service = Mock()
        mock_service.get_pending_reviews.return_value = []
        mock_service_class.return_value = mock_service

        response = client.get("/api/v1/review/pending")

        assert response.status_code == 200
        result = response.json()
        assert isinstance(result["data"], list)
        assert len(result["data"]) == 0


def test_get_review_by_id_success():
    """Test retrieving a specific review by ID."""
    review_id = uuid4()
    mock_review = {
        "id": review_id,
        "item_type": "entity",
        "entity_id": uuid4(),
        "confidence_score": 0.92,
        "grounding_quote": "Found in document section 1.3",
        "source_text": "Neural networks are powerful machine learning models",
        "status": "pending",
        "priority": 8,
        "created_at": "2024-01-20T08:00:00",
        "reviewed_at": None,
        "reviewer_notes": None
    }

    with patch('knowledge_base.review_api.ReviewService') as mock_service_class:
        mock_service = Mock()
        mock_service.get_review_by_id.return_value = mock_review
        mock_service_class.return_value = mock_service

        response = client.get(f"/api/v1/review/{review_id}")

        assert response.status_code == 200
        result = response.json()
        assert UUID(result["data"]["id"]) == review_id
        assert result["data"]["status"] == "pending"
        assert result["data"]["priority"] == 8
        mock_service.get_review_by_id.assert_called_once_with(review_id)


def test_get_review_by_id_not_found():
    """Test retrieving non-existent review returns error response."""
    review_id = uuid4()

    with patch('knowledge_base.review_api.ReviewService') as mock_service_class:
        mock_service = Mock()
        mock_service.get_review_by_id.return_value = None
        mock_service_class.return_value = mock_service

        response = client.get(f"/api/v1/review/{review_id}")

        assert response.status_code == 200
        result = response.json()
        assert "error" in result
        assert "not found" in result["error"]["message"]


def test_approve_review_success():
    """Test approving a review successfully."""
    review_id = uuid4()

    with patch('knowledge_base.review_api.ReviewService') as mock_service_class:
        mock_service = Mock()
        mock_service.approve_review.return_value = True
        mock_service_class.return_value = mock_service

        approval_data = {"reviewer_notes": "Verified and approved"}
        response = client.post(
            f"/api/v1/review/{review_id}/approve",
            json=approval_data
        )

        assert response.status_code == 200
        assert response.json()["data"] is True
        mock_service.approve_review.assert_called_once_with(
            review_id,
            "Verified and approved"
        )


def test_approve_review_not_found():
    """Test approving non-existent review returns error response."""
    review_id = uuid4()

    with patch('knowledge_base.review_api.ReviewService') as mock_service_class:
        mock_service = Mock()
        mock_service.approve_review.return_value = False
        mock_service_class.return_value = mock_service

        response = client.post(
            f"/api/v1/review/{review_id}/approve",
            json={}
        )

        assert response.status_code == 200
        result = response.json()
        assert "error" in result
        assert "not found" in result["error"]["message"]


def test_approve_review_without_notes():
    """Test approving a review without reviewer notes."""
    review_id = uuid4()

    with patch('knowledge_base.review_api.ReviewService') as mock_service_class:
        mock_service = Mock()
        mock_service.approve_review.return_value = True
        mock_service_class.return_value = mock_service

        response = client.post(
            f"/api/v1/review/{review_id}/approve",
            json={}
        )

        assert response.status_code == 200
        assert response.json()["data"] is True
        mock_service.approve_review.assert_called_once_with(review_id, None)


def test_reject_review_success():
    """Test rejecting a review with corrections."""
    review_id = uuid4()
    corrections = {
        "entity_name": "Corrected Name",
        "entity_type": "PERSON",
        "confidence_score": 0.95
    }

    with patch('knowledge_base.review_api.ReviewService') as mock_service_class:
        mock_service = Mock()
        mock_service.reject_review.return_value = True
        mock_service_class.return_value = mock_service

        rejection_data = {
            "corrections": corrections,
            "reviewer_notes": "Name corrected based on context"
        }
        response = client.post(
            f"/api/v1/review/{review_id}/reject",
            json=rejection_data
        )

        assert response.status_code == 200
        assert response.json()["data"] is True
        mock_service.reject_review.assert_called_once_with(
            review_id,
            corrections,
            "Name corrected based on context"
        )


def test_reject_review_not_found():
    """Test rejecting non-existent review returns error response."""
    review_id = uuid4()

    with patch('knowledge_base.review_api.ReviewService') as mock_service_class:
        mock_service = Mock()
        mock_service.reject_review.return_value = False
        mock_service_class.return_value = mock_service

        response = client.post(
            f"/api/v1/review/{review_id}/reject",
            json={"corrections": {}}
        )

        assert response.status_code == 200
        result = response.json()
        assert "error" in result
        assert "not found" in result["error"]["message"]


def test_reject_review_minimal_corrections():
    """Test rejecting review with minimal corrections."""
    review_id = uuid4()

    with patch('knowledge_base.review_api.ReviewService') as mock_service_class:
        mock_service = Mock()
        mock_service.reject_review.return_value = True
        mock_service_class.return_value = mock_service

        rejection_data = {"corrections": {"confidence_score": 0.5}}
        response = client.post(
            f"/api/v1/review/{review_id}/reject",
            json=rejection_data
        )

        assert response.status_code == 200
        assert response.json()["data"] is True
        mock_service.reject_review.assert_called_once_with(
            review_id,
            {"confidence_score": 0.5},
            None
        )


def test_review_end_to_end_workflow():
    """Test complete review workflow: retrieve, approve, verify."""
    review_id = uuid4()
    mock_review = {
        "id": review_id,
        "item_type": "entity",
        "entity_id": uuid4(),
        "confidence_score": 0.88,
        "status": "pending",
        "priority": 6,
        "created_at": "2024-01-20T12:00:00"
    }

    with patch('knowledge_base.review_api.ReviewService') as mock_service_class:
        mock_service = Mock()
        mock_service.get_review_by_id.return_value = mock_review
        mock_service.approve_review.return_value = True
        mock_service_class.return_value = mock_service

        # Step 1: Get review details
        response = client.get(f"/api/v1/review/{review_id}")
        assert response.status_code == 200
        assert response.json()["data"]["status"] == "pending"

        # Step 2: Approve the review
        approve_response = client.post(
            f"/api/v1/review/{review_id}/approve",
            json={"reviewer_notes": "Approved after review"}
        )
        assert approve_response.status_code == 200
        assert approve_response.json()["data"] is True

        mock_service.get_review_by_id.assert_called_once_with(review_id)
        mock_service.approve_review.assert_called_once()


def test_filter_reviews_by_priority():
    """Test filtering reviews by priority threshold."""
    high_priority_reviews = [
        {
            "id": uuid4(),
            "item_type": "entity",
            "confidence_score": 0.95,
            "status": "pending",
            "priority": 9,
            "created_at": "2024-01-20T10:00:00"
        }
    ]

    with patch('knowledge_base.review_api.ReviewService') as mock_service_class:
        mock_service = Mock()
        mock_service.get_pending_reviews.return_value = high_priority_reviews
        mock_service_class.return_value = mock_service

        response = client.get("/api/v1/review/pending?limit=100")
        assert response.status_code == 200

        result = response.json()
        high_priority = [r for r in result["data"] if r["priority"] >= 8]
        assert len(high_priority) > 0


def test_search_in_review_queue():
    """Test searching for specific reviews in the queue."""
    mock_reviews = [
        {
            "id": uuid4(),
            "item_type": "document",
            "source_text": "Important findings about neural networks",
            "confidence_score": 0.91,
            "status": "pending",
            "priority": 7,
            "created_at": "2024-01-19T09:00:00"
        }
    ]

    with patch('knowledge_base.review_api.ReviewService') as mock_service_class:
        mock_service = Mock()
        mock_service.get_pending_reviews.return_value = mock_reviews
        mock_service_class.return_value = mock_service

        response = client.get("/api/v1/review/pending")
        assert response.status_code == 200

        result = response.json()
        # Simulate search by filtering results
        matches = [r for r in result["data"] if "neural" in r.get("source_text", "").lower()]
        assert len(matches) == 1
