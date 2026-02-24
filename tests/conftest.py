"""Test fixtures for KBV2 tests."""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from knowledge_base.main import app


@pytest.fixture
def client():
    """FastAPI TestClient fixture for testing the API."""
    return TestClient(app)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return {
        "id": "test-doc-001",
        "name": "Bitcoin ETF Analysis",
        "content": """
        BlackRock's IBIT ETF has reached $45 billion in AUM since its launch.
        The fund has seen consistent daily inflows, with over $500 million added yesterday.
        Fidelity's FBTC has also performed well, though with lower overall assets.
        Both ETFs track the spot price of Bitcoin and hold actual BTC in custody.
        """,
        "metadata": {
            "source": "test",
            "domain": "INSTITUTIONAL_CRYPTO",
        },
    }


@pytest.fixture
def sample_entity():
    """Sample entity for testing."""
    return {
        "id": "test-entity-001",
        "name": "BlackRock IBIT",
        "type": "ETF",
        "properties": {
            "aum": "$45 billion",
            "underlying": "Bitcoin",
        },
    }


@pytest.fixture
def sample_edge():
    """Sample edge/relationship for testing."""
    return {
        "id": "test-edge-001",
        "source_id": "test-entity-001",
        "target_id": "test-entity-002",
        "type": "tracks",
        "properties": {},
    }


@pytest.fixture
def sample_chunk():
    """Sample chunk for testing."""
    return {
        "id": "test-chunk-001",
        "document_id": "test-doc-001",
        "text": "BlackRock's IBIT ETF has reached $45 billion in AUM.",
        "chunk_index": 0,
        "metadata": {},
    }
