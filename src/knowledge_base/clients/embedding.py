"""Ollama-based embedding client for vector generation.

This module provides an async client for generating embeddings using Ollama's
embedding API with the bge-m3 model (1024 dimensions).
"""

import asyncio
import logging
from typing import Optional

import httpx
import numpy as np

from knowledge_base.config.loader import get_config

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Async client for generating embeddings via Ollama.

    Uses the bge-m3 model for high-quality multilingual embeddings.
    Supports batch embedding for efficient processing of multiple texts.

    Example:
        >>> client = EmbeddingClient()
        >>> embedding = await client.embed("Bitcoin ETF inflows...")
        >>> embeddings = await client.embed_batch(["text1", "text2"])
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        """Initialize the embedding client.

        Args:
            api_base: Ollama API base URL. Uses config default if not provided.
            model: Embedding model name. Uses config default if not provided.
            dimension: Embedding dimension. Uses config default if not provided.
        """
        config = get_config()
        self._api_base = api_base or config.embedding.api_base
        self._model = model or config.embedding.model
        self._dimension = dimension or config.embedding.dimension
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """Initialize the HTTP client and verify model availability."""
        self._client = httpx.AsyncClient(
            base_url=self._api_base,
            timeout=httpx.Timeout(60.0),
        )

        # Verify model is available
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            if self._model not in model_names:
                logger.warning(
                    f"Model {self._model} not found in Ollama. "
                    f"Available: {model_names}. Run: ollama pull {self._model}"
                )
            else:
                logger.info(f"Embedding client initialized with model {self._model}")
        except Exception as e:
            logger.warning(f"Could not verify Ollama model availability: {e}")

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector (list of floats).
        """
        if not self._client:
            await self.initialize()

        response = await self._client.post(
            "/api/embeddings",
            json={
                "model": self._model,
                "prompt": text,
            },
        )
        response.raise_for_status()

        result = response.json()
        embedding = result.get("embedding", [])

        if len(embedding) != self._dimension:
            logger.warning(
                f"Embedding dimension mismatch: expected {self._dimension}, got {len(embedding)}"
            )

        return embedding

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 10,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Processes texts in batches to avoid overwhelming the API.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process concurrently.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        if not self._client:
            await self.initialize()

        embeddings = []

        # Process in batches to avoid overwhelming Ollama
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Create tasks for concurrent processing
            tasks = [self.embed(text) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks, return_exceptions=True)

            for j, emb in enumerate(batch_embeddings):
                if isinstance(emb, Exception):
                    logger.error(f"Failed to embed text {i + j}: {emb}")
                    # Use zero vector as fallback
                    embeddings.append([0.0] * self._dimension)
                else:
                    embeddings.append(emb)

        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query.

        This is an alias for embed() for compatibility with search pipelines.

        Args:
            query: Search query text.

        Returns:
            Query embedding vector.
        """
        return await self.embed(query)

    def get_dimension(self) -> int:
        """Get the embedding dimension.

        Returns:
            Embedding dimension.
        """
        return self._dimension

    async def __aenter__(self) -> "EmbeddingClient":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
