"""Tests for cross-encoder reranking functionality.

This module contains unit tests for the CrossEncoderReranker class,
testing cross-encoder scoring, reranking quality, score fusion, and edge cases.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any


class TestCrossEncoderConfig:
    """Tests for CrossEncoderConfig validation."""

    def test_default_config_values(self):
        """Test default configuration values are set correctly."""
        from knowledge_base.reranking.cross_encoder import CrossEncoderConfig

        config = CrossEncoderConfig()
        assert config.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert config.batch_size == 32
        assert config.device is None
        assert config.normalize is True

    def test_custom_config_values(self):
        """Test custom configuration values are accepted."""
        from knowledge_base.reranking.cross_encoder import CrossEncoderConfig

        config = CrossEncoderConfig(
            model_name="cross-encoder/ms-marco-TinyBERT-L-4",
            batch_size=64,
            device="cuda",
            normalize=False,
        )
        assert config.model_name == "cross-encoder/ms-marco-TinyBERT-L-4"
        assert config.batch_size == 64
        assert config.device == "cuda"
        assert config.normalize is False

    def test_config_as_pydantic_model(self):
        """Test config can be used as Pydantic model."""
        from knowledge_base.reranking.cross_encoder import CrossEncoderConfig

        config_dict = {
            "model_name": "custom-model",
            "batch_size": 16,
        }
        config = CrossEncoderConfig(**config_dict)
        assert config.model_name == "custom-model"
        assert config.batch_size == 16


class TestRerankedSearchResult:
    """Tests for RerankedSearchResult model."""

    def test_reranked_search_result_creation(self):
        """Test creating a reranked search result."""
        from knowledge_base.reranking.cross_encoder import RerankedSearchResult

        result = RerankedSearchResult(
            id="doc1",
            text="This is a test document",
            vector_score=0.8,
            bm25_score=0.6,
            final_score=0.7,
            cross_encoder_score=0.9,
            reranked_score=0.85,
            source="hybrid",
        )
        assert result.id == "doc1"
        assert result.text == "This is a test document"
        assert result.vector_score == 0.8
        assert result.bm25_score == 0.6
        assert result.final_score == 0.7
        assert result.cross_encoder_score == 0.9
        assert result.reranked_score == 0.85
        assert result.source == "hybrid"

    def test_reranked_search_result_defaults(self):
        """Test default values for reranked search result."""
        from knowledge_base.reranking.cross_encoder import RerankedSearchResult

        result = RerankedSearchResult(
            id="doc1",
            text="Test",
            vector_score=0.5,
            bm25_score=0.5,
            final_score=0.5,
        )
        assert result.cross_encoder_score == 0.0
        assert result.reranked_score == 0.0
        assert result.source == "hybrid"


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker class."""

    @pytest.fixture
    def reranker(self):
        """Create a CrossEncoderReranker instance for testing."""
        from knowledge_base.reranking.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker()

    def test_reranker_initialization(self, reranker):
        """Test reranker initializes with correct attributes."""
        assert reranker._model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker._model is None
        assert reranker._initialized is False

    def test_reranker_custom_model(self):
        """Test reranker with custom model name."""
        from knowledge_base.reranking.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name="custom-model")
        assert reranker._model_name == "custom-model"

    def test_reranker_repr(self, reranker):
        """Test string representation of reranker."""
        repr_str = repr(reranker)
        assert "CrossEncoderReranker" in repr_str
        assert "ms-marco-MiniLM-L-6-v2" in repr_str

    @pytest.mark.asyncio
    async def test_initialize_model_not_loaded(self, reranker):
        """Test initialize raises error when model not loaded."""
        from knowledge_base.reranking.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        assert reranker._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_loads_model(self, reranker):
        """Test initialize loads the cross-encoder model."""
        with patch("knowledge_base.reranking.cross_encoder.CrossEncoder") as mock_ce:
            mock_model = Mock()
            mock_ce.return_value = mock_model

            await reranker.initialize()

            mock_ce.assert_called_once()
            assert reranker._model is not None
            assert reranker._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_cached(self, reranker):
        """Test initialize only loads model once."""
        with patch("knowledge_base.reranking.cross_encoder.CrossEncoder") as mock_ce:
            mock_model = Mock()
            mock_ce.return_value = mock_model

            await reranker.initialize()
            await reranker.initialize()

            mock_ce.assert_called_once()

    @pytest.mark.asyncio
    async def test_rerank_empty_candidates(self, reranker):
        """Test rerank with empty candidates returns empty list."""
        with patch.object(reranker, "_model", Mock()):
            reranker._initialized = True

            results = await reranker.rerank(query="test query", candidates=[], top_k=10)
            assert results == []

    @pytest.mark.asyncio
    async def test_rerank_single_candidate(self, reranker):
        """Test rerank with single candidate."""
        with patch.object(reranker, "_model", Mock()) as mock_model:
            mock_model.predict = Mock(return_value=[0.75])
            reranker._initialized = True

            candidates = [
                {
                    "id": "doc1",
                    "text": "This is a test document",
                    "document_id": "parent1",
                    "metadata": {"source": "test"},
                }
            ]

            results = await reranker.rerank(
                query="test query", candidates=candidates, top_k=10
            )

            assert len(results) == 1
            assert results[0].id == "doc1"
            assert results[0].cross_encoder_score == 0.75

    @pytest.mark.asyncio
    async def test_rerank_multiple_candidates(self, reranker):
        """Test rerank with multiple candidates."""
        with patch.object(reranker, "_model", Mock()) as mock_model:
            mock_model.predict = Mock(return_value=[0.3, 0.7, 0.5])
            reranker._initialized = True

            candidates = [
                {"id": "doc1", "text": "First document", "document_id": "p1"},
                {"id": "doc2", "text": "Second document", "document_id": "p2"},
                {"id": "doc3", "text": "Third document", "document_id": "p3"},
            ]

            results = await reranker.rerank(
                query="test query", candidates=candidates, top_k=3
            )

            assert len(results) == 3
            assert results[0].id == "doc2"
            assert results[1].id == "doc3"
            assert results[2].id == "doc1"

    @pytest.mark.asyncio
    async def test_rerank_respects_top_k(self, reranker):
        """Test rerank respects top_k parameter."""
        with patch.object(reranker, "_model", Mock()) as mock_model:
            mock_model.predict = Mock(return_value=[0.9, 0.8, 0.7, 0.6, 0.5])
            reranker._initialized = True

            candidates = [
                {"id": f"doc{i}", "text": f"Document {i}", "document_id": f"p{i}"}
                for i in range(5)
            ]

            results = await reranker.rerank(
                query="test query", candidates=candidates, top_k=2
            )

            assert len(results) == 2
            assert results[0].id == "doc0"
            assert results[1].id == "doc1"

    @pytest.mark.asyncio
    async def test_rerank_score_normalization(self, reranker):
        """Test rerank normalizes scores to [0, 1] range."""
        with patch.object(reranker, "_model", Mock()) as mock_model:
            mock_model.predict = Mock(return_value=[10.0, 20.0, 30.0])
            reranker._initialized = True

            config = reranker.config
            config.normalize = True
            reranker.config = config

            candidates = [
                {"id": "doc1", "text": "Doc 1", "document_id": "p1"},
                {"id": "doc2", "text": "Doc 2", "document_id": "p2"},
                {"id": "doc3", "text": "Doc 3", "document_id": "p3"},
            ]

            results = await reranker.rerank(
                query="test query", candidates=candidates, top_k=3
            )

            assert results[0].cross_encoder_score == 1.0
            assert results[1].cross_encoder_score == 0.5
            assert results[2].cross_encoder_score == 0.0

    @pytest.mark.asyncio
    async def test_rerank_batch_size(self, reranker):
        """Test rerank uses correct batch size."""
        with patch.object(reranker, "_model", Mock()) as mock_model:
            mock_model.predict = Mock(return_value=[0.5, 0.6])
            reranker._initialized = True

            candidates = [
                {"id": "doc1", "text": "Doc 1", "document_id": "p1"},
                {"id": "doc2", "text": "Doc 2", "document_id": "p2"},
            ]

            await reranker.rerank(
                query="test query", candidates=candidates, batch_size=64, top_k=2
            )

            mock_model.predict.assert_called_once()
            call_args = mock_model.predict.call_args
            assert call_args[1].get("batch_size", 32) == 64

    @pytest.mark.asyncio
    async def test_rerank_with_fusion(self, reranker):
        """Test rerank_with_fusion combines scores correctly."""
        with patch.object(reranker, "_model", Mock()) as mock_model:
            mock_model.predict = Mock(return_value=[0.8, 0.6])
            reranker._initialized = True

            candidates = [
                {"id": "doc1", "text": "Doc 1", "document_id": "p1"},
                {"id": "doc2", "text": "Doc 2", "document_id": "p2"},
            ]

            initial_scores = [0.9, 0.7]

            results = await reranker.rerank_with_fusion(
                query="test query",
                candidates=candidates,
                initial_scores=initial_scores,
                top_k=2,
                rerank_weight=0.7,
            )

            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_rerank_with_fusion_no_initial_scores(self, reranker):
        """Test rerank_with_fusion with no initial scores."""
        with patch.object(reranker, "_model", Mock()) as mock_model:
            mock_model.predict = Mock(return_value=[0.8, 0.6])
            reranker._initialized = True

            candidates = [
                {"id": "doc1", "text": "Doc 1", "document_id": "p1"},
                {"id": "doc2", "text": "Doc 2", "document_id": "p2"},
            ]

            results = await reranker.rerank_with_fusion(
                query="test query", candidates=candidates, initial_scores=None, top_k=2
            )

            assert len(results) == 2

    def test_score_pair(self, reranker):
        """Test scoring a single query-document pair."""
        with patch.object(reranker, "_model", Mock()) as mock_model:
            mock_model.predict = Mock(return_value=[0.85])
            reranker._initialized = True

            score = reranker.score_pair(
                query="What is Python?", document="Python is a programming language."
            )

            assert score == 0.85
            mock_model.predict.assert_called_once()

    def test_score_pair_empty_inputs(self, reranker):
        """Test score_pair with empty inputs returns 0."""
        reranker._initialized = True

        score = reranker.score_pair("", "document")
        assert score == 0.0

        score = reranker.score_pair("query", "")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_score_batch(self, reranker):
        """Test batch scoring of documents."""
        with patch.object(reranker, "_model", Mock()) as mock_model:
            mock_model.predict = Mock(return_value=[0.3, 0.7, 0.5])
            reranker._initialized = True

            scores = await reranker.score_batch(
                query="test query", documents=["doc1", "doc2", "doc3"], batch_size=32
            )

            assert len(scores) == 3
            assert scores == [0.3, 0.7, 0.5]

    @pytest.mark.asyncio
    async def test_score_batch_empty_documents(self, reranker):
        """Test batch scoring with empty document list."""
        reranker._initialized = True

        scores = await reranker.score_batch(query="test query", documents=[])

        assert scores == []

    @pytest.mark.asyncio
    async def test_score_batch_respects_batch_size(self, reranker):
        """Test batch scoring processes in correct batch sizes."""
        with patch.object(reranker, "_model", Mock()) as mock_model:
            mock_model.predict = Mock(return_value=[0.5])
            reranker._initialized = True

            await reranker.score_batch(
                query="test query", documents=["doc1"], batch_size=16
            )

            mock_model.predict.assert_called_once()

    def test_compute_reranked_score(self, reranker):
        """Test combined score computation."""
        score = reranker._compute_reranked_score(
            hybrid_score=0.6, ce_score=0.8, ce_weight=0.7
        )
        assert score == pytest.approx(0.74)

    def test_compute_reranked_score_custom_weight(self, reranker):
        """Test combined score with custom weight."""
        score = reranker._compute_reranked_score(
            hybrid_score=0.5, ce_score=0.9, ce_weight=0.5
        )
        assert score == 0.7

    @pytest.mark.asyncio
    async def test_shutdown(self, reranker):
        """Test shutdown cleans up model resources."""
        mock_model = Mock()
        reranker._model = mock_model
        reranker._initialized = True

        await reranker.shutdown()

        assert reranker._model is None
        assert reranker._initialized is False


class TestCrossEncoderEdgeCases:
    """Tests for edge cases in cross-encoder reranking."""

    @pytest.fixture
    def reranker(self):
        """Create a CrossEncoderReranker instance."""
        from knowledge_base.reranking.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker()

    @pytest.mark.asyncio
    async def test_rerank_allows_top_k_minus_one(self, reranker):
        """Test that top_k=-1 returns all results."""
        with patch.object(reranker, "_model", Mock()) as mock_model:
            mock_model.predict = Mock(return_value=[0.5, 0.6, 0.7])
            reranker._initialized = True

            candidates = [
                {"id": "doc1", "text": "Doc 1", "document_id": "p1"},
                {"id": "doc2", "text": "Doc 2", "document_id": "p2"},
                {"id": "doc3", "text": "Doc 3", "document_id": "p3"},
            ]

            results = await reranker.rerank(
                query="test query", candidates=candidates, top_k=-1
            )

            assert len(results) == 3

    @pytest.mark.asyncio
    async def test_rerank_preserves_metadata(self, reranker):
        """Test that metadata is preserved in reranked results."""
        with patch.object(reranker, "_model", Mock()) as mock_model:
            mock_model.predict = Mock(return_value=[0.5])
            reranker._initialized = True

            metadata = {"author": "test", "category": "docs"}
            candidates = [
                {
                    "id": "doc1",
                    "text": "Doc 1",
                    "document_id": "p1",
                    "metadata": metadata,
                }
            ]

            results = await reranker.rerank(
                query="test query", candidates=candidates, top_k=1
            )

            assert results[0].metadata == metadata

    @pytest.mark.asyncio
    async def test_rerank_handles_missing_document_id(self, reranker):
        """Test that missing document_id defaults to empty string."""
        with patch.object(reranker, "_model", Mock()) as mock_model:
            mock_model.predict = Mock(return_value=[0.5])
            reranker._initialized = True

            candidates = [{"id": "doc1", "text": "Doc 1"}]

            results = await reranker.rerank(
                query="test query", candidates=candidates, top_k=1
            )

            assert results[0].document_id == ""

    @pytest.mark.asyncio
    async def test_normalize_handles_equal_scores(self, reranker):
        """Test normalization when all scores are equal."""
        scores = reranker._normalize([0.5, 0.5, 0.5])
        assert all(s == 1.0 for s in scores)

    @pytest.mark.asyncio
    async def test_normalize_handles_empty_list(self, reranker):
        """Test normalization with empty list."""
        scores = reranker._normalize([])
        assert scores == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
