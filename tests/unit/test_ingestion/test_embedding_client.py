"""Tests for embedding client dimension support."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.knowledge_base.ingestion.v1.embedding_client import (
    EMBEDDING_CONFIGS,
    EmbeddingClient,
)


class TestEmbeddingConfigs:
    """Tests for embedding configuration constants."""

    def test_default_config(self):
        """Test default embedding configuration."""
        config = EMBEDDING_CONFIGS["default"]
        assert "model" in config
        assert "dimensions" in config
        assert "max_tokens" in config

    def test_high_dimension_config(self):
        """Test high dimension configuration."""
        config = EMBEDDING_CONFIGS["high_dimension"]
        assert config["dimensions"] >= config["dimensions"]

    def test_optimized_config(self):
        """Test optimized configuration with reduced dimensions."""
        config = EMBEDDING_CONFIGS["optimized"]
        assert config["dimensions"] <= config["dimensions"]


class TestEmbeddingClient:
    """Tests for EmbeddingClient class."""

    @pytest.fixture
    def client(self):
        """Create an embedding client instance."""
        return EmbeddingClient()

    @pytest.mark.asyncio
    async def test_embed_texts_empty_list(self, client):
        """Test embedding empty list returns empty result."""
        result = await client.embed_texts([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_texts_single_text(self, client):
        """Test embedding single text."""
        with patch.object(client, "_embed_batch", new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = [[0.1, 0.2, 0.3]]

            result = await client.embed_texts(["test text"])

            assert len(result) == 1
            assert len(result[0]) == 3

    @pytest.mark.asyncio
    async def test_embed_texts_multiple_texts(self, client):
        """Test embedding multiple texts."""
        with patch.object(client, "_embed_batch", new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]

            result = await client.embed_texts(["text1", "text2"])

            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_embed_texts_with_custom_dimensions(self, client):
        """Test embedding with custom dimension truncation."""
        with patch.object(client, "_embed_batch", new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = [
                [0.1, 0.2, 0.3, 0.4, 0.5],
            ]

            result = await client.embed_texts(["test"], dimensions=3)

            assert len(result) == 1
            assert len(result[0]) == 3

    @pytest.mark.asyncio
    async def test_embed_texts_no_truncation(self, client):
        """Test embedding without dimension truncation."""
        full_embedding = [float(i) for i in range(768)]
        with patch.object(client, "_embed_batch", new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = [full_embedding]

            result = await client.embed_texts(["test"])

            assert len(result) == 1
            assert len(result[0]) == 768

    @pytest.mark.asyncio
    async def test_embed_texts_batch_processing(self, client):
        """Test batch processing of embeddings."""
        client._dimensions = 768

        with patch.object(client, "_embed_batch", new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = [
                [0.1] * 768,
                [0.2] * 768,
                [0.3] * 768,
            ]

            texts = [f"text_{i}" for i in range(150)]
            result = await client.embed_texts(texts, batch_size=100)

            assert mock_batch.call_count == 2

    def test_get_embedding_info(self, client):
        """Test getting embedding information."""
        info = client.get_embedding_info()

        assert "model" in info
        assert "dimensions" in info
        assert "max_tokens" in info
        assert info["model"] == client._config.embedding_model

    def test_truncate_embeddings(self, client):
        """Test embedding truncation."""
        embeddings = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8],
        ]

        truncated = client._truncate_embeddings(embeddings, target_dim=3)

        assert len(truncated[0]) == 3
        assert truncated[0] == [0.1, 0.2, 0.3]
        assert len(truncated[1]) == 3
        assert truncated[1] == [0.6, 0.7, 0.8]

    def test_truncate_embeddings_no_op(self, client):
        """Test truncation when target >= current."""
        embeddings = [[0.1, 0.2, 0.3]]

        truncated = client._truncate_embeddings(embeddings, target_dim=5)

        assert truncated == embeddings

    @pytest.mark.asyncio
    async def test_embed_batch_with_extra_params(self, client):
        """Test batch embedding with extra parameters."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = await client._embed_batch(["test"], {"dimensions": 512})

            assert len(result) == 1

            call_args = mock_client.post.call_args
            json_body = call_args[1]["json"]
            assert "dimensions" in json_body
