"""Tests for reranking pipeline functionality.

This module contains unit tests for the RerankingPipeline class,
testing the complete search pipeline, score fusion, and integration
with hybrid search and cross-encoder components.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any


class TestRerankingPipelineConfig:
    """Tests for RerankingPipelineConfig validation."""

    def test_default_config_values(self):
        """Test default configuration values are set correctly."""
        from knowledge_base.reranking.reranking_pipeline import RerankingPipelineConfig

        config = RerankingPipelineConfig()
        assert config.initial_top_k == 50
        assert config.final_top_k == 10
        assert config.rerank_weight == 0.7
        assert config.use_reranking is True

    def test_custom_config_values(self):
        """Test custom configuration values are accepted."""
        from knowledge_base.reranking.reranking_pipeline import RerankingPipelineConfig

        config = RerankingPipelineConfig(
            initial_top_k=100, final_top_k=20, rerank_weight=0.8, use_reranking=False
        )
        assert config.initial_top_k == 100
        assert config.final_top_k == 20
        assert config.rerank_weight == 0.8
        assert config.use_reranking is False


class TestRerankedSearchResultWithExplanation:
    """Tests for RerankedSearchResultWithExplanation model."""

    def test_explained_result_creation(self):
        """Test creating a reranked result with explanation."""
        from knowledge_base.reranking.reranking_pipeline import (
            RerankedSearchResultWithExplanation,
        )

        result = RerankedSearchResultWithExplanation(
            id="doc1",
            text="This is a test document",
            vector_score=0.8,
            bm25_score=0.6,
            final_score=0.7,
            cross_encoder_score=0.9,
            reranked_score=0.85,
            source="hybrid",
            explanation="Strong semantic match",
            rank_factors={"vector": 0.8, "cross_encoder": 0.9},
        )
        assert result.id == "doc1"
        assert result.explanation == "Strong semantic match"
        assert result.rank_factors["vector"] == 0.8

    def test_explained_result_defaults(self):
        """Test default values for explained result."""
        from knowledge_base.reranking.reranking_pipeline import (
            RerankedSearchResultWithExplanation,
        )

        result = RerankedSearchResultWithExplanation(
            id="doc1",
            text="Test",
            vector_score=0.5,
            bm25_score=0.5,
            final_score=0.5,
            reranked_score=0.5,
        )
        assert result.cross_encoder_score == 0.0
        assert result.explanation == ""
        assert result.rank_factors == {}


class TestRerankingPipeline:
    """Tests for RerankingPipeline class."""

    @pytest.fixture
    def mock_hybrid_search(self):
        """Create a mock hybrid search engine."""
        hybrid = Mock()
        hybrid.search = AsyncMock()
        return hybrid

    @pytest.fixture
    def mock_cross_encoder(self):
        """Create a mock cross-encoder reranker."""
        ce = Mock()
        ce.rerank = AsyncMock()
        ce.rerank_with_scores = AsyncMock()
        ce.is_initialized = True
        return ce

    @pytest.fixture
    def pipeline(self, mock_hybrid_search, mock_cross_encoder):
        """Create a RerankingPipeline instance for testing."""
        from knowledge_base.reranking.reranking_pipeline import RerankingPipeline

        return RerankingPipeline(
            hybrid_search=mock_hybrid_search, cross_encoder=mock_cross_encoder
        )

    def test_pipeline_initialization(
        self, pipeline, mock_hybrid_search, mock_cross_encoder
    ):
        """Test pipeline initializes with correct components."""
        assert pipeline._hybrid is mock_hybrid_search
        assert pipeline._cross_encoder is mock_cross_encoder

    def test_pipeline_initialization_with_rrf(
        self, mock_hybrid_search, mock_cross_encoder
    ):
        """Test pipeline with RRF fuser."""
        from knowledge_base.reranking.reranking_pipeline import RerankingPipeline
        from knowledge_base.reranking.rrf_fuser import ReciprocalRankFuser

        fuser = ReciprocalRankFuser()
        pipeline = RerankingPipeline(
            hybrid_search=mock_hybrid_search,
            cross_encoder=mock_cross_encoder,
            rr_fuser=fuser,
        )
        assert pipeline._rr_fuser is fuser

    def test_pipeline_initialization_requires_hybrid(self):
        """Test pipeline raises error without hybrid search."""
        from knowledge_base.reranking.reranking_pipeline import RerankingPipeline

        with pytest.raises(ValueError):
            RerankingPipeline(hybrid_search=None, cross_encoder=Mock())

    def test_pipeline_initialization_requires_cross_encoder(self, mock_hybrid_search):
        """Test pipeline raises error without cross encoder."""
        from knowledge_base.reranking.reranking_pipeline import RerankingPipeline

        with pytest.raises(ValueError):
            RerankingPipeline(hybrid_search=mock_hybrid_search, cross_encoder=None)

    @pytest.mark.asyncio
    async def test_search_returns_hybrid_results_when_disabled(
        self, pipeline, mock_hybrid_search
    ):
        """Test search returns hybrid results when reranking is disabled."""
        mock_results = [
            Mock(
                id="doc1", text="Doc 1", final_score=0.8, source="hybrid", metadata={}
            ),
            Mock(
                id="doc2", text="Doc 2", final_score=0.7, source="hybrid", metadata={}
            ),
        ]
        mock_hybrid_search.search = AsyncMock(return_value=mock_results)

        pipeline.config.use_reranking = False

        results = await pipeline.search(
            query="test query", initial_top_k=50, final_top_k=10
        )

        mock_hybrid_search.search.assert_called_once()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_calls_hybrid_with_correct_params(
        self, pipeline, mock_hybrid_search
    ):
        """Test search passes correct parameters to hybrid search."""
        mock_results = []
        mock_hybrid_search.search = AsyncMock(return_value=mock_results)

        await pipeline.search(
            query="machine learning",
            initial_top_k=50,
            final_top_k=10,
            vector_weight=0.6,
            bm25_weight=0.4,
            filters={"category": "docs"},
            domain="tech",
        )

        mock_hybrid_search.search.assert_called_once_with(
            query="machine learning",
            vector_weight=0.6,
            bm25_weight=0.4,
            top_k=50,
            filters={"category": "docs"},
            domain="tech",
        )

    @pytest.mark.asyncio
    async def test_search_reranks_when_enabled(
        self, pipeline, mock_hybrid_search, mock_cross_encoder
    ):
        """Test search calls cross-encoder when reranking is enabled."""
        mock_candidates = [
            Mock(
                id="doc1", text="Doc 1", final_score=0.8, source="hybrid", metadata={}
            ),
            Mock(
                id="doc2", text="Doc 2", final_score=0.7, source="hybrid", metadata={}
            ),
        ]
        mock_hybrid_search.search = AsyncMock(return_value=mock_candidates)

        mock_reranked = [
            Mock(
                id="doc2",
                text="Doc 2",
                cross_encoder_score=0.9,
                reranked_score=0.85,
                metadata={},
            ),
            Mock(
                id="doc1",
                text="Doc 1",
                cross_encoder_score=0.8,
                reranked_score=0.80,
                metadata={},
            ),
        ]
        mock_cross_encoder.rerank = AsyncMock(return_value=mock_reranked)

        results = await pipeline.search(
            query="test query", initial_top_k=50, final_top_k=10
        )

        mock_cross_encoder.rerank.assert_called_once()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_no_candidates(
        self, pipeline, mock_hybrid_search
    ):
        """Test search returns empty list when hybrid returns no results."""
        mock_hybrid_search.search = AsyncMock(return_value=[])

        results = await pipeline.search(
            query="test query", initial_top_k=50, final_top_k=10
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_validates_initial_top_k(self, pipeline):
        """Test search validates initial_top_k >= final_top_k."""
        with pytest.raises(ValueError):
            await pipeline.search(query="test query", initial_top_k=5, final_top_k=10)

    @pytest.mark.asyncio
    async def test_search_validates_weights(self, pipeline):
        """Test search validates vector_weight + bm25_weight = 1.0."""
        with pytest.raises(ValueError):
            await pipeline.search(
                query="test query",
                initial_top_k=50,
                final_top_k=10,
                vector_weight=0.6,
                bm25_weight=0.3,
            )

    @pytest.mark.asyncio
    async def test_search_validates_final_top_k_positive(self, pipeline):
        """Test search validates final_top_k is positive."""
        with pytest.raises(ValueError):
            await pipeline.search(query="test query", initial_top_k=50, final_top_k=0)


class TestRerankingPipelineSearchWithExplanation:
    """Tests for search_with_explanation method."""

    @pytest.fixture
    def mock_hybrid_search(self):
        """Create a mock hybrid search engine."""
        hybrid = Mock()
        hybrid.search = AsyncMock()
        return hybrid

    @pytest.fixture
    def mock_cross_encoder(self):
        """Create a mock cross-encoder reranker."""
        ce = Mock()
        ce.rerank_with_scores = AsyncMock()
        ce.is_initialized = True
        return ce

    @pytest.fixture
    def pipeline(self, mock_hybrid_search, mock_cross_encoder):
        """Create a RerankingPipeline instance."""
        from knowledge_base.reranking.reranking_pipeline import RerankingPipeline

        return RerankingPipeline(
            hybrid_search=mock_hybrid_search, cross_encoder=mock_cross_encoder
        )

    @pytest.mark.asyncio
    async def test_search_with_explanation_returns_explained_results(
        self, pipeline, mock_hybrid_search, mock_cross_encoder
    ):
        """Test search_with_explanation returns explained results."""
        mock_candidates = [
            Mock(
                id="doc1",
                text="Doc 1",
                vector_score=0.8,
                bm25_score=0.6,
                final_score=0.7,
                source="hybrid",
                metadata={},
            ),
        ]
        mock_hybrid_search.search = AsyncMock(return_value=mock_candidates)

        mock_reranked = [
            Mock(
                id="doc1",
                text="Doc 1",
                cross_encoder_score=0.9,
                reranked_score=0.85,
                metadata={},
            ),
        ]
        mock_cross_encoder.rerank_with_scores = AsyncMock(
            return_value=(mock_reranked, [0.9])
        )

        results = await pipeline.search_with_explanation(
            query="test query", initial_top_k=50, final_top_k=10
        )

        assert len(results) == 1
        assert results[0].explanation != ""
        assert "vector_score" in results[0].rank_factors

    @pytest.mark.asyncio
    async def test_search_with_explanation_initializes_cross_encoder(
        self, pipeline, mock_hybrid_search, mock_cross_encoder
    ):
        """Test search_with_explanation initializes cross-encoder if needed."""
        mock_cross_encoder.is_initialized = False
        mock_cross_encoder.initialize = AsyncMock()

        mock_candidates = []
        mock_hybrid_search.search = AsyncMock(return_value=mock_candidates)

        await pipeline.search_with_explanation(
            query="test query", initial_top_k=50, final_top_k=10
        )

        mock_cross_encoder.initialize.assert_called_once()


class TestRerankingPipelineSearchWithRRF:
    """Tests for search_with_rrf method."""

    @pytest.fixture
    def mock_hybrid_search(self):
        """Create a mock hybrid search engine."""
        hybrid = Mock()
        hybrid.search = AsyncMock()
        return hybrid

    @pytest.fixture
    def mock_cross_encoder(self):
        """Create a mock cross-encoder reranker."""
        ce = Mock()
        ce.rerank = AsyncMock()
        ce.is_initialized = True
        return ce

    @pytest.fixture
    def pipeline(self, mock_hybrid_search, mock_cross_encoder):
        """Create a RerankingPipeline instance with RRF fuser."""
        from knowledge_base.reranking.reranking_pipeline import RerankingPipeline
        from knowledge_base.reranking.rrf_fuser import ReciprocalRankFuser

        return RerankingPipeline(
            hybrid_search=mock_hybrid_search,
            cross_encoder=mock_cross_encoder,
            rr_fuser=ReciprocalRankFuser(),
        )

    @pytest.mark.asyncio
    async def test_search_with_rrf_validates_non_empty_queries(self, pipeline):
        """Test search_with_rrf validates non-empty queries list."""
        with pytest.raises(ValueError):
            await pipeline.search_with_rrf(queries=[])

    @pytest.mark.asyncio
    async def test_search_with_rrf_executes_for_each_query(
        self, pipeline, mock_hybrid_search, mock_cross_encoder
    ):
        """Test search_with_rrf executes pipeline for each query."""
        mock_candidates = [
            Mock(
                id="doc1", text="Doc 1", final_score=0.8, source="hybrid", metadata={}
            ),
        ]
        mock_hybrid_search.search = AsyncMock(return_value=mock_candidates)

        mock_reranked = [
            Mock(id="doc1", text="Doc 1", reranked_score=0.8, metadata={}),
        ]
        mock_cross_encoder.rerank = AsyncMock(return_value=mock_reranked)

        queries = ["query1", "query2"]

        await pipeline.search_with_rrf(
            queries=queries, initial_top_k=50, final_top_k=10
        )

        assert mock_hybrid_search.search.call_count == 2
        assert mock_cross_encoder.rerank.call_count == 2


class TestRerankingPipelineHealthCheck:
    """Tests for health_check method."""

    @pytest.fixture
    def mock_cross_encoder(self):
        """Create a mock cross-encoder reranker."""
        ce = Mock()
        ce.is_initialized = True
        ce.model_name = "test-model"
        return ce

    @pytest.fixture
    def pipeline(self, mock_cross_encoder):
        """Create a RerankingPipeline instance."""
        from knowledge_base.reranking.reranking_pipeline import RerankingPipeline
        from knowledge_base.reranking.rrf_fuser import ReciprocalRankFuser

        return RerankingPipeline(
            hybrid_search=Mock(),
            cross_encoder=mock_cross_encoder,
            rr_fuser=ReciprocalRankFuser(),
        )

    def test_health_check_returns_component_status(self, pipeline, mock_cross_encoder):
        """Test health_check returns status of all components."""
        health = pipeline.health_check()

        assert "cross_encoder_healthy" in health
        assert "cross_encoder_model" in health
        assert "rr_fuser_config" in health
        assert "overall_healthy" in health

        assert health["cross_encoder_healthy"] is True
        assert health["cross_encoder_model"] == "test-model"

    def test_health_check_returns_model_when_healthy(self, pipeline):
        """Test health_check returns model name when initialized."""
        health = pipeline.health_check()
        assert health["cross_encoder_model"] == "test-model"

    def test_health_check_returns_none_when_not_healthy(self, pipeline):
        """Test health_check returns None for model when not initialized."""
        pipeline._cross_encoder.is_initialized = False

        health = pipeline.health_check()
        assert health["cross_encoder_healthy"] is False
        assert health["cross_encoder_model"] is None


class TestRerankingPipelineEdgeCases:
    """Tests for edge cases in reranking pipeline."""

    @pytest.fixture
    def mock_hybrid_search(self):
        """Create a mock hybrid search engine."""
        hybrid = Mock()
        hybrid.search = AsyncMock()
        return hybrid

    @pytest.fixture
    def mock_cross_encoder(self):
        """Create a mock cross-encoder reranker."""
        ce = Mock()
        ce.rerank = AsyncMock()
        ce.is_initialized = True
        return ce

    @pytest.fixture
    def pipeline(self, mock_hybrid_search, mock_cross_encoder):
        """Create a RerankingPipeline instance."""
        from knowledge_base.reranking.reranking_pipeline import RerankingPipeline

        return RerankingPipeline(
            hybrid_search=mock_hybrid_search, cross_encoder=mock_cross_encoder
        )

    @pytest.mark.asyncio
    async def test_search_handles_candidates_less_than_final_k(
        self, pipeline, mock_hybrid_search
    ):
        """Test search returns all candidates when fewer than final_k."""
        mock_results = [
            Mock(
                id="doc1", text="Doc 1", final_score=0.8, source="hybrid", metadata={}
            ),
        ]
        mock_hybrid_search.search = AsyncMock(return_value=mock_results)

        results = await pipeline.search(
            query="test query", initial_top_k=50, final_top_k=10
        )

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_logs_info(
        self, pipeline, mock_hybrid_search, mock_cross_encoder
    ):
        """Test search logs appropriate information."""
        mock_results = []
        mock_hybrid_search.search = AsyncMock(return_value=mock_results)

        import logging

        with patch("knowledge_base.reranking.reranking_pipeline.logger") as mock_logger:
            await pipeline.search(query="test query", initial_top_k=50, final_top_k=10)

            assert mock_logger.info.called

    def test_pipeline_repr(self, pipeline):
        """Test string representation of pipeline."""
        repr_str = repr(pipeline)
        assert "RerankingPipeline" in repr_str

    @pytest.mark.asyncio
    async def test_generate_explanation_with_high_scores(self, pipeline):
        """Test explanation generation with high scores."""
        mock_candidate = Mock()
        mock_candidate.vector_score = 0.8
        mock_candidate.bm25_score = 0.7
        mock_candidate.source = "hybrid"

        explanation = pipeline._generate_explanation(
            query="test query",
            candidate=mock_candidate,
            ce_score=0.9,
            hybrid_score=0.75,
        )

        assert (
            "strong semantic match" in explanation
            or "high cross-encoder" in explanation
        )

    @pytest.mark.asyncio
    async def test_generate_explanation_with_low_scores(self, pipeline):
        """Test explanation generation with low scores."""
        mock_candidate = Mock()
        mock_candidate.vector_score = 0.2
        mock_candidate.bm25_score = 0.1
        mock_candidate.source = "bm25"

        explanation = pipeline._generate_explanation(
            query="test query",
            candidate=mock_candidate,
            ce_score=0.1,
            hybrid_score=0.15,
        )

        assert "moderate relevance" in explanation or "low cross-encoder" in explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
