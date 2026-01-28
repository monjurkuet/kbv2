import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field
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
    embedding_model: str = "nomic-embed-text"
    dimensions: int = 2048  # Database dimension size (supports bge-m3=1024 and OpenAI up to 3072)

    @classmethod
    def for_model(
        cls, model_name: str, url: str = "http://localhost:11434"
    ) -> "EmbeddingConfig":
        dimension_map = {
            "nomic-embed-text": 768,
            "nomic-embed-text-v1.5": 768,
            "gte-large": 1024,
            "e5-large-v2": 1024,
            "bge-large-en-v1.5": 1024,
            "bge-m3": 1024,  # Primary model - multilingual support
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return cls(
            embedding_url=url,
            embedding_model=model_name,
            # Use database dimension size (2048) not model dimension size
            # The embedding will be truncated or padded as needed
            dimensions=2048,
        )


EMBEDDING_CONFIGS: Dict[str, Dict[str, Any]] = {
    "default": {"model": "nomic-embed-text", "dimensions": 768, "max_tokens": 8191},
    "openai_small": {
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "max_tokens": 8191,
    },
    "openai_large": {
        "model": "text-embedding-3-large",
        "dimensions": 3072,  # Will be truncated to 2048 in database
        "max_tokens": 8191,
    },
    "high_dimension": {
        "model": "text-embedding-3-large",
        "dimensions": 3072,  # Will be truncated to 2048 in database
        "max_tokens": 8191,
    },
    "bge_m3": {"model": "bge-m3", "dimensions": 1024, "max_tokens": 8191},  # Recommended for multilingual
    "optimized": {"model": "nomic-embed-text", "dimensions": 512, "max_tokens": 8191},
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
            "prompt": text,
        }
        if extra_params:
            request_body.update(extra_params)

        response = await client.post(
            f"{self._config.embedding_url}/api/embeddings",
            headers={"Content-Type": "application/json"},
            json=request_body,
        )
        response.raise_for_status()
        data = response.json()

        embedding = data.get("embedding", [])
        if not embedding:
            raise ValueError(f"No embedding in response for text: {text[:50]}...")

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
        """Embed multiple texts using Ollama.

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding vectors.
        """
        client = await self._get_client()
        all_embeddings: list[list[float]] = []

        # Ollama processes one text per request (or batch)
        headers = {
            "Content-Type": "application/json",
        }

        for text in texts:
            try:
                embedding = await self._embed_with_retry(text)
                
                # Truncate or pad to database dimension size (2048)
                if len(embedding) > self._dimensions:
                    embedding = embedding[:self._dimensions]
                elif len(embedding) < self._dimensions:
                    embedding = embedding + [0.0] * (self._dimensions - len(embedding))
                
                all_embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to embed text: {e}")
                all_embeddings.append([0.0] * self._dimensions)

        return all_embeddings

    async def embed_texts(
        self,
        texts: List[str],
        dimensions: Optional[int] = None,
        batch_size: int = 100,
    ) -> List[List[float]]:
        """Embed multiple texts with optional dimension truncation.

        Processes a list of texts in batches and returns their embedding vectors.
        Supports dimension truncation for optimized storage when full precision
        is not required.

        Args:
            texts: List of texts to embed.
            dimensions: Target embedding dimensions. If less than native
                dimensions, embeddings will be truncated. If None, uses
                the configured default dimensions.
            batch_size: Number of texts to process per API call.

        Returns:
            List of embedding vectors, each as a list of floats.
        """
        if not texts:
            return []

        target_dim = dimensions or self._dimensions

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await self.embed_batch(batch)

            if target_dim < self._dimensions and batch_embeddings:
                all_embeddings.extend(
                    self._truncate_embeddings(batch_embeddings, target_dim)
                )
            else:
                # Pad or truncate to exactly database dimension size (2048)
                for emb in batch_embeddings:
                    if len(emb) < self._dimensions:
                        emb = emb + [0.0] * (self._dimensions - len(emb))
                    elif len(emb) > self._dimensions:
                        emb = emb[:self._dimensions]
                    all_embeddings.append(emb)

        return all_embeddings

    def _truncate_embeddings(
        self,
        embeddings: List[List[float]],
        target_dim: int,
    ) -> List[List[float]]:
        """Truncate embeddings to target dimensions.

        Args:
            embeddings: List of embedding vectors.
            target_dim: Target number of dimensions.

        Returns:
            List of truncated embedding vectors.
        """
        truncated: List[List[float]] = []
        for emb in embeddings:
            if len(emb) > target_dim:
                truncated.append(emb[:target_dim])
            else:
                truncated.append(emb)
        return truncated

    async def _embed_batch(
        self,
        texts: List[str],
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[List[float]]:
        """Embed a batch of texts with optional parameters.

        Args:
            texts: Texts to embed.
            extra_params: Additional parameters for the API call.

        Returns:
            List of embedding vectors.
        """
        all_embeddings: List[List[float]] = []

        for text in texts:
            try:
                embedding = await self._embed_with_retry(text, extra_params)
                
                # Ensure exact dimension size for database
                if len(embedding) < self._dimensions:
                    embedding = embedding + [0.0] * (self._dimensions - len(embedding))
                elif len(embedding) > self._dimensions:
                    embedding = embedding[:self._dimensions]
                
                all_embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to embed text after retries: {e}")
                # Return zero vector of correct dimension
                all_embeddings.append([0.0] * self._dimensions)

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
