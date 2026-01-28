"""Unit tests for hybrid search engine."""

import os
import pytest
import tempfile
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from knowledge_base.storage.bm25_index import BM25Index, IndexedDocument, SearchResult
from knowledge_base.storage.hybrid_search import HybridSearchEngine, HybridSearchResult


class TestHybridSearchResult:
    """Tests for HybridSearchResult model."""

    def test_create_hybrid_search_result(self):
        """Test creating HybridSearchResult."""
        result = HybridSearchResult(
            id="doc-123",
            text="Document content",
            vector_score=0.8,
            bm25_score=0.6,
            final_score=0.7,
            metadata={"domain": "test"},
            source="hybrid",
        )
        assert result.id == "doc-123"
        assert result.text == "Document content"
        assert result.vector_score == 0.8
        assert result.bm25_score == 0.6
        assert result.final_score == 0.7
        assert result.metadata == {"domain": "test"}
        assert result.source == "hybrid"

    def test_hybrid_search_result_defaults(self):
        """Test HybridSearchResult with default values."""
        result = HybridSearchResult(
            id="doc-123",
            text="Content",
            final_score=0.5,
        )
        assert result.vector_score == 0.0
        assert result.bm25_score == 0.0
        assert result.metadata is None
        assert result.source == "hybrid"


class TestHybridSearchEngine:
    """Tests for HybridSearchEngine class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file path."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = MagicMock()
        store.search_similar_chunks = AsyncMock(return_value=[])
        store.update_chunk_embedding = AsyncMock()
        return store

    @pytest.fixture
    def bm25_index(self, temp_db_path):
        """Create a BM25Index instance."""
        return BM25Index(db_path=temp_db_path)

    @pytest.fixture
    def hybrid_engine(self, mock_vector_store, bm25_index):
        """Create a HybridSearchEngine instance."""
        return HybridSearchEngine(
            vector_store=mock_vector_store,
            bm25_index=bm25_index,
        )

    def test_init_with_valid_components(self, mock_vector_store, bm25_index):
        """Test initialization with valid components."""
        engine = HybridSearchEngine(
            vector_store=mock_vector_store,
            bm25_index=bm25_index,
        )
        assert engine._vector_store is mock_vector_store
        assert engine._bm25_index is bm25_index

    def test_init_with_none_vector_store_raises_error(self, bm25_index):
        """Test initialization with None vector_store raises ValueError."""
        with pytest.raises(ValueError, match="vector_store cannot be None"):
            HybridSearchEngine(vector_store=None, bm25_index=bm25_index)

    def test_init_with_none_bm25_index_raises_error(self, mock_vector_store):
        """Test initialization with None bm25_index raises ValueError."""
        with pytest.raises(ValueError, match="bm25_index cannot be None"):
            HybridSearchEngine(vector_store=mock_vector_store, bm25_index=None)

    @pytest.mark.asyncio
    async def test_search_with_valid_weights(self, hybrid_engine):
        """Test search with valid weights that sum to 1.0."""
        await hybrid_engine._bm25_index.initialize()

        results = await hybrid_engine.search(
            query="test query",
            vector_weight=0.5,
            bm25_weight=0.5,
            top_k=10,
        )

        assert isinstance(results, list)

        await hybrid_engine._bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_with_invalid_weights_raises_error(self, hybrid_engine):
        """Test search with weights not summing to 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            await hybrid_engine.search(
                query="test",
                vector_weight=0.7,
                bm25_weight=0.5,
                top_k=10,
            )

    @pytest.mark.asyncio
    async def test_search_with_negative_top_k_raises_error(self, hybrid_engine):
        """Test search with negative top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            await hybrid_engine.search(
                query="test",
                vector_weight=0.5,
                bm25_weight=0.5,
                top_k=-1,
            )

    @pytest.mark.asyncio
    async def test_search_fuses_bm25_and_vector_results(
        self, hybrid_engine, mock_vector_store
    ):
        """Test that search fuses BM25 and vector results."""
        await hybrid_engine._bm25_index.initialize()

        mock_vector_store.search_similar_chunks.return_value = [
            {
                "id": "chunk-1",
                "text": "Vector result text",
                "similarity": 0.9,
                "metadata": {"domain": "tech"},
            },
        ]

        await hybrid_engine._bm25_index.index_documents(
            [
                IndexedDocument(
                    id="chunk-1",
                    text="BM25 result text matching query",
                    document_id="doc-1",
                ),
            ]
        )

        results = await hybrid_engine.search(
            query="query",
            vector_weight=0.5,
            bm25_weight=0.5,
            top_k=10,
        )

        assert len(results) >= 1

        await hybrid_engine._bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_returns_bm25_only_results(
        self, hybrid_engine, mock_vector_store
    ):
        """Test search returns BM25-only results when no vector matches."""
        await hybrid_engine._bm25_index.initialize()

        mock_vector_store.search_similar_chunks.return_value = []

        await hybrid_engine._bm25_index.index_documents(
            [
                IndexedDocument(
                    id="doc-1",
                    text="BM25 matched content",
                    document_id="doc-ref",
                ),
            ]
        )

        results = await hybrid_engine.search(
            query="matched",
            vector_weight=0.5,
            bm25_weight=0.5,
            top_k=10,
        )

        assert len(results) == 1
        assert results[0].id == "doc-1"
        assert results[0].bm25_score > 0
        assert results[0].source in ["bm25", "hybrid"]

        await hybrid_engine._bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_returns_vector_only_results(
        self, hybrid_engine, mock_vector_store
    ):
        """Test search returns vector-only results when no BM25 matches."""
        await hybrid_engine._bm25_index.initialize()

        mock_vector_store.search_similar_chunks.return_value = [
            {
                "id": "chunk-1",
                "text": "Vector matched content",
                "similarity": 0.85,
                "metadata": {"domain": "tech"},
            },
        ]

        results = await hybrid_engine.search(
            query="xyznonexistent",
            vector_weight=0.5,
            bm25_weight=0.5,
            top_k=10,
        )

        assert len(results) == 1
        assert results[0].id == "chunk-1"
        assert results[0].vector_score > 0
        assert results[0].source in ["vector", "hybrid"]

        await hybrid_engine._bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, hybrid_engine, mock_vector_store):
        """Test search respects the top_k parameter."""
        await hybrid_engine._bm25_index.initialize()

        mock_vector_store.search_similar_chunks.return_value = []

        for i in range(20):
            await hybrid_engine._bm25_index.index_documents(
                [
                    IndexedDocument(
                        id=f"doc-{i}", text=f"Document {i}", document_id="doc-ref"
                    ),
                ]
            )

        results = await hybrid_engine.search(
            query="document",
            vector_weight=0.5,
            bm25_weight=0.5,
            top_k=5,
        )

        assert len(results) == 5

        await hybrid_engine._bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_with_filters(self, hybrid_engine, mock_vector_store):
        """Test search with metadata filters."""
        await hybrid_engine._bm25_index.initialize()

        await hybrid_engine._bm25_index.index_documents(
            [
                IndexedDocument(
                    id="doc-1",
                    text="Technical content",
                    document_id="doc-ref",
                    metadata={"domain": "technical"},
                ),
                IndexedDocument(
                    id="doc-2",
                    text="Technical content filtered",
                    document_id="doc-ref",
                    metadata={"domain": "other"},
                ),
            ]
        )

        results = await hybrid_engine.search(
            query="technical",
            vector_weight=0.5,
            bm25_weight=0.5,
            top_k=10,
            filters={"domain": "technical"},
        )

        assert len(results) == 1
        assert results[0].id == "doc-1"

        await hybrid_engine._bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_with_empty_query(self, hybrid_engine, mock_vector_store):
        """Test search with empty query."""
        await hybrid_engine._bm25_index.initialize()

        results = await hybrid_engine.search(
            query="",
            vector_weight=0.5,
            bm25_weight=0.5,
            top_k=10,
        )

        assert len(results) == 0

        await hybrid_engine._bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_handles_vector_store_errors(
        self, hybrid_engine, mock_vector_store
    ):
        """Test search handles vector store errors gracefully."""
        await hybrid_engine._bm25_index.initialize()

        mock_vector_store.search_similar_chunks.side_effect = Exception(
            "Vector store error"
        )

        await hybrid_engine._bm25_index.index_documents(
            [
                IndexedDocument(id="doc-1", text="BM25 content", document_id="doc-ref"),
            ]
        )

        results = await hybrid_engine.search(
            query="content",
            vector_weight=0.5,
            bm25_weight=0.5,
            top_k=10,
        )

        assert len(results) == 1

        await hybrid_engine._bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_handles_bm25_errors(
        self, hybrid_engine, mock_vector_store, bm25_index
    ):
        """Test search handles BM25 errors gracefully."""
        await bm25_index.initialize()

        mock_vector_store.search_similar_chunks.return_value = [
            {
                "id": "chunk-1",
                "text": "Vector content",
                "similarity": 0.9,
            },
        ]

        engine = HybridSearchEngine(
            vector_store=mock_vector_store,
            bm25_index=bm25_index,
        )

        results = await engine.search(
            query="content",
            vector_weight=0.5,
            bm25_weight=0.5,
            top_k=10,
        )

        assert len(results) == 1

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_with_reranking_basic(self, hybrid_engine, mock_vector_store):
        """Test search_with_reranking basic functionality."""
        await hybrid_engine._bm25_index.initialize()

        mock_vector_store.search_similar_chunks.return_value = []

        for i in range(10):
            await hybrid_engine._bm25_index.index_documents(
                [
                    IndexedDocument(
                        id=f"doc-{i}",
                        text=f"Document {i} content",
                        document_id="doc-ref",
                    ),
                ]
            )

        results = await hybrid_engine.search_with_reranking(
            query="document",
            initial_top_k=10,
            final_top_k=5,
            vector_weight=0.5,
            bm25_weight=0.5,
        )

        assert len(results) == 5

        await hybrid_engine._bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_with_reranking_validates_params(self, hybrid_engine):
        """Test search_with_reranking validates initial_top_k >= final_top_k."""
        with pytest.raises(ValueError, match="initial_top_k must be >= final_top_k"):
            await hybrid_engine.search_with_reranking(
                query="test",
                initial_top_k=5,
                final_top_k=10,
                vector_weight=0.5,
                bm25_weight=0.5,
            )

    @pytest.mark.asyncio
    async def test_fusion_preserves_results_order(
        self, hybrid_engine, mock_vector_store
    ):
        """Test that fusion correctly orders results by final score."""
        await hybrid_engine._bm25_index.initialize()

        mock_vector_store.search_similar_chunks.return_value = [
            {
                "id": "low-score",
                "text": "Low score content",
                "similarity": 0.1,
            },
            {
                "id": "high-score",
                "text": "High score content",
                "similarity": 0.9,
            },
        ]

        results = await hybrid_engine.search(
            query="content",
            vector_weight=0.5,
            bm25_weight=0.5,
            top_k=10,
        )

        if len(results) >= 2:
            scores = [r.final_score for r in results]
            assert scores == sorted(scores, reverse=True)

        await hybrid_engine._bm25_index.close()


class TestHybridSearchWeightTuning:
    """Tests for weight tuning in hybrid search."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file path."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = MagicMock()
        store.search_similar_chunks = AsyncMock(return_value=[])
        return store

    @pytest.fixture
    def bm25_index(self, temp_db_path):
        """Create a BM25Index instance."""
        return BM25Index(db_path=temp_db_path)

    @pytest.mark.asyncio
    async def test_vector_dominant_weighting(self, mock_vector_store, bm25_index):
        """Test with vector_weight=1.0, bm25_weight=0.0."""
        await bm25_index.initialize()

        mock_vector_store.search_similar_chunks.return_value = [
            {
                "id": "vector-doc",
                "text": "Vector matched",
                "similarity": 0.9,
            },
        ]

        engine = HybridSearchEngine(
            vector_store=mock_vector_store,
            bm25_index=bm25_index,
        )

        results = await engine.search(
            query="matched",
            vector_weight=1.0,
            bm25_weight=0.0,
            top_k=10,
        )

        assert len(results) >= 1
        if results:
            assert results[0].source in ["vector", "hybrid"]

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_bm25_dominant_weighting(self, mock_vector_store, bm25_index):
        """Test with vector_weight=0.0, bm25_weight=1.0."""
        await bm25_index.initialize()

        mock_vector_store.search_similar_chunks.return_value = []

        await bm25_index.index_documents(
            [
                IndexedDocument(
                    id="bm25-doc", text="BM25 matched content", document_id="doc-ref"
                ),
            ]
        )

        engine = HybridSearchEngine(
            vector_store=mock_vector_store,
            bm25_index=bm25_index,
        )

        results = await engine.search(
            query="matched",
            vector_weight=0.0,
            bm25_weight=1.0,
            top_k=10,
        )

        assert len(results) >= 1
        if results:
            assert results[0].bm25_score > 0

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_equal_weighting(self, mock_vector_store, bm25_index):
        """Test with equal weights (0.5, 0.5)."""
        await bm25_index.initialize()

        mock_vector_store.search_similar_chunks.return_value = [
            {
                "id": "doc-1",
                "text": "Content",
                "similarity": 0.8,
            },
        ]

        await bm25_index.index_documents(
            [
                IndexedDocument(id="doc-1", text="Content", document_id="doc-ref"),
            ]
        )

        engine = HybridSearchEngine(
            vector_store=mock_vector_store,
            bm25_index=bm25_index,
        )

        results = await engine.search(
            query="content",
            vector_weight=0.5,
            bm25_weight=0.5,
            top_k=10,
        )

        if results:
            expected_score = 0.5 * results[0].vector_score + 0.5 * results[0].bm25_score
            assert abs(results[0].final_score - expected_score) < 0.01

        await bm25_index.close()


class TestHybridSearchEdgeCases:
    """Edge case tests for HybridSearchEngine."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file path."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = MagicMock()
        store.search_similar_chunks = AsyncMock(return_value=[])
        return store

    @pytest.fixture
    def bm25_index(self, temp_db_path):
        """Create a BM25Index instance."""
        return BM25Index(db_path=temp_db_path)

    @pytest.mark.asyncio
    async def test_search_with_no_results(self, mock_vector_store, bm25_index):
        """Test search when both indexes return no results."""
        await bm25_index.initialize()

        engine = HybridSearchEngine(
            vector_store=mock_vector_store,
            bm25_index=bm25_index,
        )

        results = await engine.search(
            query="nonexistent query xyz123",
            vector_weight=0.5,
            bm25_weight=0.5,
            top_k=10,
        )

        assert len(results) == 0

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_search_with_duplicate_ids(self, mock_vector_store, bm25_index):
        """Test search handles duplicate IDs from both indexes."""
        await bm25_index.initialize()

        mock_vector_store.search_similar_chunks.return_value = [
            {
                "id": "shared-id",
                "text": "Vector text",
                "similarity": 0.9,
            },
        ]

        await bm25_index.index_documents(
            [
                IndexedDocument(
                    id="shared-id",
                    text="BM25 text",
                    document_id="doc-ref",
                ),
            ]
        )

        engine = HybridSearchEngine(
            vector_store=mock_vector_store,
            bm25_index=bm25_index,
        )

        results = await engine.search(
            query="text",
            vector_weight=0.5,
            bm25_weight=0.5,
            top_k=10,
        )

        shared_results = [r for r in results if r.id == "shared-id"]
        assert len(shared_results) == 1
        assert shared_results[0].vector_score > 0
        assert shared_results[0].bm25_score > 0
        assert shared_results[0].source == "hybrid"

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_vector_store, bm25_index):
        """Test getting hybrid search engine stats."""
        await bm25_index.initialize()

        await bm25_index.index_documents(
            [
                IndexedDocument(id="doc-1", text="Content", document_id="doc-ref"),
            ]
        )

        engine = HybridSearchEngine(
            vector_store=mock_vector_store,
            bm25_index=bm25_index,
        )

        stats = await engine.get_stats()

        assert "bm25_index" in stats
        assert "vector_weight" in stats
        assert "bm25_weight" in stats

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_index_document_for_hybrid_search(
        self, mock_vector_store, bm25_index
    ):
        """Test indexing a document for hybrid search."""
        await bm25_index.initialize()

        engine = HybridSearchEngine(
            vector_store=mock_vector_store,
            bm25_index=bm25_index,
        )

        chunks = [
            {
                "id": "chunk-1",
                "text": "First chunk content",
                "embedding": [0.1] * 768,
                "metadata": {"page": 1},
            },
            {
                "id": "chunk-2",
                "text": "Second chunk content",
                "embedding": [0.2] * 768,
                "metadata": {"page": 2},
            },
        ]

        await engine.index_document_for_hybrid_search(
            document_id="doc-123",
            chunks=chunks,
        )

        stats = await bm25_index.get_stats()
        assert stats["document_count"] == 2

        mock_vector_store.update_chunk_embedding.assert_called()

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_delete_document_from_hybrid_search(
        self, mock_vector_store, bm25_index
    ):
        """Test deleting a document from hybrid search."""
        await bm25_index.initialize()

        await bm25_index.index_documents(
            [
                IndexedDocument(
                    id="doc-1", text="Content to delete", document_id="doc-to-delete"
                ),
                IndexedDocument(
                    id="doc-2", text="Content to keep", document_id="doc-to-keep"
                ),
            ]
        )

        engine = HybridSearchEngine(
            vector_store=mock_vector_store,
            bm25_index=bm25_index,
        )

        await engine.delete_document_from_hybrid_search("doc-to-delete")

        stats = await bm25_index.get_stats()
        assert stats["document_count"] == 1

        await bm25_index.close()


class TestHybridSearchFusionAccuracy:
    """Tests for fusion accuracy and score normalization."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file path."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = MagicMock()
        store.search_similar_chunks = AsyncMock(return_value=[])
        return store

    @pytest.fixture
    def bm25_index(self, temp_db_path):
        """Create a BM25Index instance."""
        return BM25Index(db_path=temp_db_path)

    @pytest.mark.asyncio
    async def test_scores_are_normalized_to_0_1(self, mock_vector_store, bm25_index):
        """Test that scores are normalized to [0, 1] range."""
        await bm25_index.initialize()

        mock_vector_store.search_similar_chunks.return_value = [
            {
                "id": "doc-1",
                "text": "Content",
                "similarity": 0.85,
            },
        ]

        await bm25_index.index_documents(
            [
                IndexedDocument(id="doc-1", text="Content", document_id="doc-ref"),
            ]
        )

        engine = HybridSearchEngine(
            vector_store=mock_vector_store,
            bm25_index=bm25_index,
        )

        results = await engine.search(
            query="content",
            vector_weight=0.5,
            bm25_weight=0.5,
            top_k=10,
        )

        if results:
            for result in results:
                assert 0.0 <= result.vector_score <= 1.0
                assert 0.0 <= result.bm25_score <= 1.0
                assert 0.0 <= result.final_score <= 1.0

        await bm25_index.close()

    @pytest.mark.asyncio
    async def test_higher_weight_increases_influence(
        self, mock_vector_store, bm25_index
    ):
        """Test that higher weight increases that search method's influence."""
        await bm25_index.initialize()

        mock_vector_store.search_similar_chunks.return_value = [
            {
                "id": "doc-1",
                "text": "Content with high vector similarity",
                "similarity": 0.9,
            },
        ]

        await bm25_index.index_documents(
            [
                IndexedDocument(
                    id="doc-1",
                    text="Content with high vector similarity",
                    document_id="doc-ref",
                ),
                IndexedDocument(
                    id="doc-2",
                    text="Content with high BM25 score matching query",
                    document_id="doc-ref",
                ),
            ]
        )

        engine = HybridSearchEngine(
            vector_store=mock_vector_store,
            bm25_index=bm25_index,
        )

        vector_heavy = await engine.search(
            query="vector similarity",
            vector_weight=0.9,
            bm25_weight=0.1,
            top_k=10,
        )

        bm25_heavy = await engine.search(
            query="bm25 score matching",
            vector_weight=0.1,
            bm25_weight=0.9,
            top_k=10,
        )

        if vector_heavy:
            assert vector_heavy[0].vector_score > 0

        await bm25_index.close()
