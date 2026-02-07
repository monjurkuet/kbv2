"""Async embedding client using OpenAI SDK.

Uses AsyncOpenAI with Ollama-compatible endpoint for bge-m3 embeddings.

Environment Variables:
- EMBEDDING_API_BASE: API base URL (default: http://localhost:11434/v1)
- EMBEDDING_API_KEY: API key (default: sk-dummy)
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_EMBEDDING_MODEL = "bge-m3"
DEFAULT_EMBEDDING_DIMENSIONS = 1024


class EmbeddingConfig(BaseModel):
    """Embedding configuration for EmbeddingClient."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="ignore")

    embedding_url: str = Field(default="http://localhost:11434")
    embedding_model: str = Field(default="bge-m3")
    dimensions: int = Field(default=1024)


class EmbeddingClient:
    """Async embedding client using OpenAI SDK.

    Compatible with Ollama running bge-m3 model.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the embedding client.

        Args:
            config: EmbeddingConfig object.
            model: Embedding model name (default: bge-m3).
            dimensions: Embedding dimensions (default: 1024 for bge-m3).
            base_url: API base URL. Defaults to EMBEDDING_API_BASE env var.
            api_key: API key. Defaults to EMBEDDING_API_KEY env var.
        """
        if config:
            self.model = config.embedding_model
            self.dimensions = config.dimensions
            base_url = config.embedding_url
        else:
            self.model = model or DEFAULT_EMBEDDING_MODEL
            self.dimensions = dimensions or DEFAULT_EMBEDDING_DIMENSIONS

        self._client = AsyncOpenAI(
            base_url=base_url
            or os.getenv("EMBEDDING_API_BASE", "http://localhost:11434/v1"),
            api_key=api_key or os.getenv("EMBEDDING_API_KEY", "sk-dummy"),
        )

    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        response = await self._client.embeddings.create(
            model=self.model,
            input=[text],
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in a single API call.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        response = await self._client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [data.embedding for data in response.data]

    async def embed_texts(
        self,
        texts: List[str],
        dimensions: Optional[int] = None,
        batch_size: int = 100,
    ) -> List[List[float]]:
        """Embed multiple texts with batching.

        Args:
            texts: List of texts to embed.
            dimensions: Ignored (always uses bge-m3 with 1024 dims).
            batch_size: Number of texts per API call.

        Returns:
            List of embedding vectors (each 1024 dimensions).
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await self.embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def get_info(self) -> Dict[str, Any]:
        """Get embedding configuration info.

        Returns:
            Dict with model, dimensions, and max_tokens.
        """
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "max_tokens": 8191,
        }

    async def close(self) -> None:
        """Close the async client."""
        await self._client.close()

    async def __aenter__(self) -> "EmbeddingClient":
        """Enter async context.

        Returns:
            Self.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        await self.close()


# Singleton instance
_embedding_client: EmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient:
    """Get or create singleton embedding client.

    Returns:
        EmbeddingClient instance.
    """
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client


async def test_embedding_client() -> None:
    """Test the embedding client functionality."""
    try:
        client = EmbeddingClient()

        # Test single embedding
        embedding = await client.embed_text("Test embedding")
        print(f"✅ Single embedding: {len(embedding)} dimensions")

        # Test batch embedding
        embeddings = await client.embed_batch(["Test 1", "Test 2", "Test 3"])
        print(
            f"✅ Batch embeddings: {len(embeddings)} vectors, {len(embeddings[0])} dims"
        )

        info = client.get_info()
        print(f"✅ Model info: {info}")

        await client.close()
        print("✅ Embedding client test passed!")

    except Exception as e:
        print(f"❌ Embedding client test failed: {e}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_embedding_client())
