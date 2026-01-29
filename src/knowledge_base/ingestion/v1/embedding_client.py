import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


logger = logging.getLogger(__name__)


class EmbeddingConfig(BaseSettings):
    """Embedding configuration."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="ignore")

    embedding_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3"  # Use bge-m3 for multilingual support
    dimensions: int = 1024  # bge-m3 dimension size

    @field_validator("embedding_url")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Strip whitespace including \r and \n from URL."""
        return v.strip()

    @classmethod
    def for_model(
        cls, model_name: str = "bge-m3", url: str = "http://localhost:11434"
    ) -> "EmbeddingConfig":
        """Create config for bge-m3 model with 1024 dimensions."""
        # Always use bge-m3 with 1024 dimensions
        if model_name != "bge-m3":
            logger.warning(f"Model {model_name} not supported, using bge-m3")
            model_name = "bge-m3"

        return cls(
            embedding_url=url,
            embedding_model=model_name,
            # Use 1024 dimensions for bge-m3 (matches database schema)
            dimensions=1024,
        )


EMBEDDING_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Only bge-m3 is supported - 1024 dimensions
    "default": {"model": "bge-m3", "dimensions": 1024, "max_tokens": 8191},
    "bge_m3": {"model": "bge-m3", "dimensions": 1024, "max_tokens": 8191},
}


class EmbeddingClient:
    """Client for generating text embeddings using Ollama.

    This client provides async methods for embedding single texts or batches
    of texts using the Ollama embeddings API. Supports configurable dimension
    truncation for optimized storage and retrieval.

    Attributes:
        _config: Configuration settings for the embedding service.
        _client: Async HTTP client for API requests.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize embedding client.

        Args:
            config: Embedding configuration.
        """
        self._config = config or EmbeddingConfig()
        self._client: httpx.AsyncClient | None = None
        self._dimensions = self._config.dimensions  # Database dimension size (2048)
        self._max_tokens = self._embedding_config["max_tokens"]
        self._max_retries = 3
        self._retry_exceptions = (
            httpx.RequestError,
            httpx.TimeoutException,
            httpx.ConnectError,
        )

    @property
    def _embedding_config(self) -> dict[str, Any]:
        """Get embedding config for current model."""
        return EMBEDDING_CONFIGS.get("default")

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (httpx.RequestError, httpx.TimeoutException, httpx.ConnectError)
        ),
    )
    async def _embed_with_retry(
        self,
        text: str,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """Embed a single text with automatic retry on transient failures."""
        client = await self._get_client()
        request_body: Dict[str, Any] = {
            "model": self._config.embedding_model,
            "input": text,  # Use 'input' not 'prompt' for /api/embed endpoint
        }
        if extra_params:
            request_body.update(extra_params)

        response = await client.post(
            f"{self._config.embedding_url}/api/embed",  # Use /api/embed endpoint
            headers={"Content-Type": "application/json"},
            json=request_body,
        )
        response.raise_for_status()
        data = response.json()

        # Ollama /api/embed returns "embeddings" (plural) as an array
        embeddings = data.get("embeddings", [])
        if not embeddings or not isinstance(embeddings, list) or len(embeddings) == 0:
            raise ValueError(f"No embeddings in response for text: {text[:50]}...")

        # Get the first (and only) embedding from the array
        embedding = embeddings[0]
        if not embedding:
            raise ValueError(f"Empty embedding in response for text: {text[:50]}...")

        return [float(x) for x in embedding]

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
        """Embed multiple texts using Ollama in parallel.

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        # Use a semaphore to limit concurrency to avoid overwhelming Ollama
        semaphore = asyncio.Semaphore(10)

        async def _embed_safe(text: str) -> list[float]:
            async with semaphore:
                try:
                    # Clean text to remove non-printable characters
                    clean_text = text.replace("\r", "").replace("\n", " ").strip()
                    if not clean_text:
                        return [0.0] * self._dimensions

                    embedding = await self._embed_with_retry(clean_text)

                    # bge-m3 produces exactly 1024 dimensions
                    if len(embedding) != self._dimensions:
                        logger.warning(
                            f"Embedding dimension mismatch: expected {self._dimensions}, got {len(embedding)}"
                        )

                    return embedding
                except Exception as e:
                    logger.error(f"Failed to embed text: {e}")
                    return [0.0] * self._dimensions

        # Execute all tasks in parallel
        tasks = [_embed_safe(text) for text in texts]
        all_embeddings = await asyncio.gather(*tasks)

        return list(all_embeddings)

    async def embed_texts(
        self,
        texts: List[str],
        dimensions: Optional[int] = None,
        batch_size: int = 100,
    ) -> List[List[float]]:
        """Embed multiple texts using bge-m3.

        Processes a list of texts in batches and returns their embedding vectors.
        Always uses 1024 dimensions for bge-m3 model.

        Args:
            texts: List of texts to embed.
            dimensions: Kept for API compatibility, but always uses 1024.
            batch_size: Number of texts to process per API call.

        Returns:
            List of embedding vectors (each 1024 dimensions).
        """
        if not texts:
            return []

        # Ignore dimensions parameter, always use bge-m3 with 1024 dims
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await self.embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding configuration.

        Returns:
            Dictionary containing model, dimensions, and max tokens.
        """
        return {
            "model": self._config.embedding_model,
            "dimensions": self._dimensions,
            "max_tokens": 8191,
            "database_dimensions": self._dimensions,
        }

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


async def test_embedding():
    """Test the embedding functionality."""
    try:
        client = EmbeddingClient()
        test_text = "This is a test sentence for embedding."

        print(f"Testing embedding with text: {test_text}")
        print(f"Model: {client._config.embedding_model}")
        print(f"Database dimensions: {client._dimensions}")
        
        embedding = await client.embed_text(test_text)
        print(f"Success! Embedding vector length: {len(embedding)}")
        print(f"Sample values: {embedding[:5]}")

        await client.close()
    except Exception as e:
        print(f"Error during embedding test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_embedding())
