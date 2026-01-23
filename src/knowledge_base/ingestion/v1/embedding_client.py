"""Ollama Embeddings API client."""

import asyncio

import httpx
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseSettings):
    """Embedding configuration."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="ignore")

    url: str = "http://localhost:11434"
    model: str = "nomic-embed-text"
    batch_size: int = 100


class EmbeddingClient:
    """Client for Ollama Embeddings API."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize embedding client.

        Args:
            config: Embedding configuration.
        """
        self._config = config or EmbeddingConfig()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.

        Returns:
            Async HTTP client.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=120.0,
                follow_redirects=True,
            )
        return self._client

    async def embed_text(
        self,
        text: str,
    ) -> list[float]:
        """Embed single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Embed multiple texts in batch.

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding vectors.
        """
        client = await self._get_client()

        all_embeddings: list[list[float]] = []

        for text in texts:
            response = await client.post(
                f"{self._config.url}/api/embeddings",
                json={
                    "model": self._config.model,
                    "prompt": text,
                },
            )
            response.raise_for_status()
            data = response.json()
            all_embeddings.append(data["embedding"])

        return all_embeddings

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
