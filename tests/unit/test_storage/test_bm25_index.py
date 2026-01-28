"""Unit tests for BM25 index implementation."""

import os
import pytest
import tempfile
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from knowledge_base.storage.bm25_index import (
    BM25Index,
    IndexedDocument,
    SearchResult,
)


class TestIndexedDocument:
    """Tests for IndexedDocument model."""

    def test_create_with_defaults(self):
        """Test creating IndexedDocument with default values."""
        doc = IndexedDocument(
            text="Test document content",
            document_id="doc-123",
        )
        assert doc.id is not None
        assert doc.text == "Test document content"
        assert doc.document_id == "doc-123"
        assert doc.metadata is None

    def test_create_with_all_fields(self):
        """Test creating IndexedDocument with all fields."""
        doc_id = str(uuid4())
        doc = IndexedDocument(
            id=doc_id,
            text="Full document text",
            document_id="doc-456",
            metadata={"domain": "technical", "author": "test"},
        )
        assert doc.id == doc_id
        assert doc.text == "Full document text"
        assert doc.document_id == "doc-456"
        assert doc.metadata == {"domain": "technical", "author": "test"}


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_create_search_result(self):
        """Test creating SearchResult."""
        result = SearchResult(
            id="doc-123",
            text="Matched text content",
            score=5.5,
            metadata={"domain": "test"},
        )
        assert result.id == "doc-123"
        assert result.text == "Matched text content"
        assert result.score == 5.5
        assert result.metadata == {"domain": "test"}

    def test_search_result_defaults(self):
        """Test SearchResult with default values."""
        result = SearchResult(
            id="doc-123",
            text="Text",
            score=1.0,
        )
        assert result.metadata is None


class TestBM25Index:
    """Tests for BM25Index class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file path."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def bm25_index(self, temp_db_path):
        """Create a BM25Index instance with temp database."""
        return BM25Index(db_path=temp_db_path)

    @pytest.mark.asyncio
    async def test_initialize_creates_database(self, temp_db_path):
        """Test that initialize creates the database file and tables."""
        index = BM25Index(db_path=temp_db_path)
        await index.initialize()

        assert os.path.exists(temp_db_path)
        stats = await index.get_stats()
        assert stats["document_count"] == 0
        assert stats["indexed"] is False

        await index.close()

    @pytest.mark.asyncio
    async def test_index_documents(self, bm25_index):
        """Test indexing documents."""
        await bm25_index.initialize()

        documents = [
            IndexedDocument(
                id="doc-1",
                text="Python programming language tutorial",
                document_id="doc-ref-1",
                metadata={"domain": "programming"},
            ),
            IndexedDocument(
                id="doc-2",
                text="Machine learning algorithms and techniques",
                document_id="doc-ref-1",
                metadata={"domain": "ai"},
            ),
            IndexedDocument(
                id="doc-3",
                text="Python data science with pandas and numpy",
                document_id="doc-ref-2",
                metadata={"domain": "programming"},
            ),
        ]

        await bm25_index.index_documents(documents)

        stats = await bm25_index.get_stats()
        assert stats["document_count"] == 3
        assert stats["indexed"] is True

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_returns_relevant_results(self, bm25_index):
        """Test that search returns relevant results for a query."""
        await bm25_index.initialize()

        documents = [
            IndexedDocument(
                id="doc-1",
                text="Python programming language tutorial",
                document_id="doc-ref-1",
            ),
            IndexedDocument(
                id="doc-2",
                text="Java programming language guide",
                document_id="doc-ref-2",
            ),
            IndexedDocument(
                id="doc-3",
                text="Machine learning with Python",
                document_id="doc-ref-3",
            ),
        ]

        await bm25_index.index_documents(documents)

        results = await bm25_index.search(query="Python", top_k=10)

        assert len(results) == 2
        result_ids = [r.id for r in results]
        assert "doc-1" in result_ids
        assert "doc-3" in result_ids
        assert "doc-2" not in result_ids

        python_results = [r for r in results if r.id in ["doc-1", "doc-3"]]
        assert len(python_results) == 2

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, bm25_index):
        """Test that search respects the top_k parameter."""
        await bm25_index.initialize()

        documents = [
            IndexedDocument(
                id=f"doc-{i}", text=f"Document number {i}", document_id="doc-ref"
            )
            for i in range(20)
        ]

        await bm25_index.index_documents(documents)

        results = await bm25_index.search(query="document", top_k=5)

        assert len(results) == 5

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_with_metadata_filters(self, bm25_index):
        """Test search with metadata filters."""
        await bm25_index.initialize()

        documents = [
            IndexedDocument(
                id="doc-1",
                text="Python programming tutorial",
                document_id="doc-ref-1",
                metadata={"domain": "programming"},
            ),
            IndexedDocument(
                id="doc-2",
                text="Python snake biology",
                document_id="doc-ref-2",
                metadata={"domain": "biology"},
            ),
            IndexedDocument(
                id="doc-3",
                text="Programming in Java",
                document_id="doc-ref-3",
                metadata={"domain": "programming"},
            ),
        ]

        await bm25_index.index_documents(documents)

        results = await bm25_index.search(
            query="Python programming",
            top_k=10,
            filters={"domain": "programming"},
        )

        assert len(results) == 1
        assert results[0].id == "doc-1"

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_with_empty_query(self, bm25_index):
        """Test search with empty query returns empty results."""
        await bm25_index.initialize()

        documents = [
            IndexedDocument(id="doc-1", text="Some content", document_id="doc-ref"),
        ]

        await bm25_index.index_documents(documents)

        results = await bm25_index.search(query="", top_k=10)

        assert len(results) == 0

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_with_no_matching_terms(self, bm25_index):
        """Test search with query terms not in any document."""
        await bm25_index.initialize()

        documents = [
            IndexedDocument(
                id="doc-1", text="Python programming", document_id="doc-ref"
            ),
        ]

        await bm25_index.index_documents(documents)

        results = await bm25_index.search(query="xyzabc123", top_k=10)

        assert len(results) == 0

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_delete_documents(self, bm25_index):
        """Test deleting documents from the index."""
        await bm25_index.initialize()

        documents = [
            IndexedDocument(id="doc-1", text="Document one", document_id="doc-ref-1"),
            IndexedDocument(id="doc-2", text="Document two", document_id="doc-ref-1"),
            IndexedDocument(id="doc-3", text="Document three", document_id="doc-ref-2"),
        ]

        await bm25_index.index_documents(documents)

        await bm25_index.delete_documents(["doc-ref-1"])

        stats = await bm25_index.get_stats()
        assert stats["document_count"] == 1

        search_results = await bm25_index.search(query="document", top_k=10)
        assert len(search_results) == 1
        assert search_results[0].id == "doc-3"

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_update_document(self, bm25_index):
        """Test updating a document in the index."""
        await bm25_index.initialize()

        original_doc = IndexedDocument(
            id="doc-1",
            text="Original content",
            document_id="doc-ref",
        )

        await bm25_index.index_documents([original_doc])

        updated_doc = IndexedDocument(
            id="doc-1",
            text="Updated content with new information",
            document_id="doc-ref",
        )

        await bm25_index.update_document(updated_doc)

        results = await bm25_index.search(query="updated", top_k=10)
        assert len(results) == 1
        assert results[0].id == "doc-1"

        results_old = await bm25_index.search(query="original", top_k=10)
        assert len(results_old) == 0

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_clear_index(self, bm25_index):
        """Test clearing all documents from the index."""
        await bm25_index.initialize()

        documents = [
            IndexedDocument(id=f"doc-{i}", text=f"Document {i}", document_id="doc-ref")
            for i in range(10)
        ]

        await bm25_index.index_documents(documents)
        await bm25_index.clear()

        stats = await bm25_index.get_stats()
        assert stats["document_count"] == 0
        assert stats["indexed"] is False

        results = await bm25_index.search(query="document", top_k=10)
        assert len(results) == 0

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_before_initialization_raises_error(self):
        """Test that search before initialization raises RuntimeError."""
        index = BM25Index(db_path=":memory:")

        with pytest.raises(RuntimeError, match="not initialized"):
            await index.search(query="test")

    @pytest.mark.asyncio
    async def test_index_documents_before_initialization_raises_error(self):
        """Test that index_documents before initialization raises RuntimeError."""
        index = BM25Index(db_path=":memory:")

        with pytest.raises(RuntimeError, match="not initialized"):
            await index.index_documents(
                [IndexedDocument(id="doc-1", text="Test", document_id="doc-ref")]
            )

    @pytest.mark.asyncio
    async def test_context_manager(self, temp_db_path):
        """Test using BM25Index as async context manager."""
        async with BM25Index(db_path=temp_db_path) as index:
            assert await index.get_stats() is not None

            documents = [
                IndexedDocument(id="doc-1", text="Test content", document_id="doc-ref"),
            ]
            await index.index_documents(documents)

        stats = await index.get_stats()
        assert stats["document_count"] == 1

    @pytest.mark.asyncio
    async def test_tokenization_handles_special_characters(self, bm25_index):
        """Test that tokenization correctly handles special characters."""
        await bm25_index.initialize()

        documents = [
            IndexedDocument(
                id="doc-1",
                text="Hello, world! How are you?",
                document_id="doc-ref",
            ),
        ]

        await bm25_index.index_documents(documents)

        results = await bm25_index.search(query="hello world", top_k=10)

        assert len(results) == 1
        assert results[0].id == "doc-1"

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_case_insensitive_search(self, bm25_index):
        """Test that search is case insensitive."""
        await bm25_index.initialize()

        documents = [
            IndexedDocument(id="doc-1", text="UPPERCASE TEXT", document_id="doc-ref"),
            IndexedDocument(id="doc-2", text="lowercase text", document_id="doc-ref"),
        ]

        await bm25_index.index_documents(documents)

        results = await bm25_index.search(query="Text", top_k=10)

        assert len(results) == 2

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_with_duplicate_document_id(self, bm25_index):
        """Test that indexing with duplicate document ID replaces existing."""
        await bm25_index.initialize()

        doc1 = IndexedDocument(id="doc-1", text="First version", document_id="doc-ref")
        doc2 = IndexedDocument(id="doc-1", text="Second version", document_id="doc-ref")

        await bm25_index.index_documents([doc1])
        await bm25_index.index_documents([doc2])

        stats = await bm25_index.get_stats()
        assert stats["document_count"] == 1

        results = await bm25_index.search(query="first", top_k=10)
        assert len(results) == 0

        results = await bm25_index.search(query="second", top_k=10)
        assert len(results) == 1

        await bm25_index.close()


class TestBM25IndexEdgeCases:
    """Edge case tests for BM25Index."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file path."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def bm25_index(self, temp_db_path):
        """Create a BM25Index instance."""
        return BM25Index(db_path=temp_db_path)

    @pytest.mark.asyncio
    async def test_search_with_empty_corpus(self, bm25_index):
        """Test search on empty corpus."""
        await bm25_index.initialize()

        results = await bm25_index.search(query="test", top_k=10)

        assert len(results) == 0

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_index_empty_document_list(self, bm25_index):
        """Test indexing empty document list."""
        await bm25_index.initialize()

        await bm25_index.index_documents([])

        stats = await bm25_index.get_stats()
        assert stats["document_count"] == 0

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_delete_non_existent_documents(self, bm25_index):
        """Test deleting documents that don't exist."""
        await bm25_index.initialize()

        documents = [
            IndexedDocument(id="doc-1", text="Document one", document_id="doc-ref"),
        ]

        await bm25_index.index_documents(documents)
        await bm25_index.delete_documents(["non-existent-id"])

        stats = await bm25_index.get_stats()
        assert stats["document_count"] == 1

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_invalid_top_k_raises_error(self, bm25_index):
        """Test that negative top_k raises ValueError."""
        await bm25_index.initialize()

        with pytest.raises(ValueError, match="top_k must be positive"):
            await bm25_index.search(query="test", top_k=-1)

        with pytest.raises(ValueError, match="top_k must be positive"):
            await bm25_index.search(query="test", top_k=0)

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_results_sorted_by_score(self, bm25_index):
        """Test that search results are sorted by score descending."""
        await bm25_index.initialize()

        documents = [
            IndexedDocument(
                id="doc-1",
                text="Python programming language Python Python",
                document_id="doc-ref",
            ),
            IndexedDocument(
                id="doc-2",
                text="Python introduction",
                document_id="doc-ref",
            ),
            IndexedDocument(
                id="doc-3",
                text="A brief guide to Python",
                document_id="doc-ref",
            ),
        ]

        await bm25_index.index_documents(documents)

        results = await bm25_index.search(query="Python", top_k=10)

        assert len(results) == 3
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_index_with_unicode_content(self, bm25_index):
        """Test indexing documents with unicode content."""
        await bm25_index.initialize()

        documents = [
            IndexedDocument(
                id="doc-1",
                text="Café résumé naïve 日本語",
                document_id="doc-ref",
            ),
        ]

        await bm25_index.index_documents(documents)

        results = await bm25_index.search(query="café", top_k=10)

        assert len(results) == 1

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_index_with_long_document(self, bm25_index):
        """Test indexing a very long document."""
        await bm25_index.initialize()

        long_text = " ".join(["word"] * 10000)
        documents = [
            IndexedDocument(id="doc-1", text=long_text, document_id="doc-ref"),
        ]

        await bm25_index.index_documents(documents)

        stats = await bm25_index.get_stats()
        assert stats["document_count"] == 1

        results = await bm25_index.search(query="word", top_k=10)
        assert len(results) == 1

        await bm25_index.close()
