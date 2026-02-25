"""Tests for SQLite storage with FTS5."""

import pytest
from pathlib import Path

from knowledge_base.storage.portable.sqlite_store import (
    Document,
    Chunk,
    SQLiteStore,
    SearchResult,
)
from knowledge_base.storage.portable.config import SQLiteConfig


@pytest.fixture
def sqlite_store(tmp_path):
    """Create a SQLite store with temporary database."""
    db_path = tmp_path / "test.db"
    config = SQLiteConfig(db_path=str(db_path))
    store = SQLiteStore(config)
    return store


class TestSQLiteStore:
    """Tests for SQLiteStore class."""

    @pytest.mark.asyncio
    async def test_initialize(self, sqlite_store):
        """Test database initialization."""
        await sqlite_store.initialize()
        assert sqlite_store._initialized is True

    @pytest.mark.asyncio
    async def test_add_document(self, sqlite_store):
        """Test adding a document."""
        await sqlite_store.initialize()

        doc = Document(
            name="Test Document",
            content="This is test content about Bitcoin ETFs.",
            domain="INSTITUTIONAL_CRYPTO",
        )

        doc_id = await sqlite_store.add_document(doc)
        assert doc_id is not None
        assert doc_id == doc.id

    @pytest.mark.asyncio
    async def test_get_document(self, sqlite_store):
        """Test retrieving a document."""
        await sqlite_store.initialize()

        doc = Document(
            name="Test Document",
            content="Test content",
            domain="BITCOIN",
        )
        await sqlite_store.add_document(doc)

        retrieved = await sqlite_store.get_document(doc.id)
        assert retrieved is not None
        # get_document returns a Document object
        assert retrieved.id == doc.id
        assert retrieved.name == "Test Document"

    @pytest.mark.asyncio
    async def test_list_documents(self, sqlite_store):
        """Test listing documents."""
        await sqlite_store.initialize()

        # Add multiple documents
        for i in range(3):
            doc = Document(
                name=f"Document {i}",
                content=f"Content {i}",
            )
            await sqlite_store.add_document(doc)

        docs = await sqlite_store.list_documents()
        assert len(docs) == 3

    @pytest.mark.asyncio
    async def test_add_chunk(self, sqlite_store):
        """Test adding a chunk."""
        await sqlite_store.initialize()

        # First add a document
        doc = Document(name="Test Doc", content="Test")
        await sqlite_store.add_document(doc)

        # Add a chunk
        chunk = Chunk(
            document_id=doc.id,
            text="This is a test chunk about Bitcoin.",
            chunk_index=0,
        )

        chunk_id = await sqlite_store.add_chunk(chunk)
        assert chunk_id is not None

    @pytest.mark.asyncio
    async def test_fts_search(self, sqlite_store):
        """Test full-text search."""
        await sqlite_store.initialize()

        # Add document with chunks
        doc = Document(name="Bitcoin Report", content="Test")
        await sqlite_store.add_document(doc)

        chunk = Chunk(
            document_id=doc.id,
            text="Bitcoin price target for 2024 is $100,000 according to analysts.",
            chunk_index=0,
        )
        await sqlite_store.add_chunk(chunk)

        # Search for Bitcoin using FTS
        results = await sqlite_store.search_fts("Bitcoin price")
        assert len(results) >= 1
        assert "Bitcoin" in results[0].text


class TestDocument:
    """Tests for Document model."""

    def test_document_creation(self):
        """Test document creation with defaults."""
        doc = Document(name="Test")
        assert doc.id is not None
        assert doc.status == "pending"
        assert doc.metadata == {}

    def test_document_to_db_dict(self):
        """Test document serialization."""
        doc = Document(
            name="Test",
            content="Content",
            domain="BITCOIN",
        )
        db_dict = doc.to_db_dict()
        assert db_dict["name"] == "Test"
        assert db_dict["domain"] == "BITCOIN"


class TestChunk:
    """Tests for Chunk model."""

    def test_chunk_creation(self):
        """Test chunk creation."""
        chunk = Chunk(
            document_id="doc-123",
            text="Test chunk text",
            chunk_index=0,
        )
        assert chunk.id is not None
        assert chunk.document_id == "doc-123"

    def test_chunk_to_db_dict(self):
        """Test chunk serialization."""
        chunk = Chunk(
            document_id="doc-123",
            text="Test text",
            chunk_index=0,
        )
        db_dict = chunk.to_db_dict()
        assert db_dict["document_id"] == "doc-123"
        assert db_dict["chunk_index"] == 0
