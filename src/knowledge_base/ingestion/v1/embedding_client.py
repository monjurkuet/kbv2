import json
import logging

"""Ollama Embeddings API client."""

import json
import logging
import os

import httpx
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)


class EmbeddingConfig(BaseSettings):
    """Embedding configuration."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="ignore")

    embedding_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"


class EmbeddingClient:
    """Client for Google Embeddings API."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize embedding client.

        Args:
            config: Embedding configuration.
        """
        self._config = config or EmbeddingConfig()
        self._client: httpx.AsyncClient | None = None
        self._api_key = os.getenv("GOOGLE_API_KEY", "")

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
            response = await client.post(
                f"{self._config.embedding_url}/api/embeddings",
                headers=headers,
                json={
                    "model": self._config.embedding_model,
                    "prompt": text,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Ollama returns embedding directly
            embedding = data.get("embedding", [])
            if not embedding:
                logger.error(f"No embedding in response: {data}")
                embedding = []

            # Return raw float list - pgvector handles conversion automatically
            if isinstance(embedding, list) and len(embedding) > 0:
                try:
                    # Ensure all values are valid floats
                    float_values = [float(x) for x in embedding]
                    all_embeddings.append(float_values)
                    logger.debug(f"Embedding generated: {len(float_values)} dimensions")
                except Exception as e:
                    logger.error(f"Failed to process embedding: {e}")
                    all_embeddings.append([])
            else:
                logger.warning("No embedding for text")
                all_embeddings.append([])

            # Debug log
            if embedding:
                logger.info(
                    f"Embedding generated: {len(embedding)} dimensions, type: {type(embedding)}"
                )
                if isinstance(embedding, list) and len(embedding) > 0:
                    logger.info(f"  Sample values: {embedding[:3]}")
                elif isinstance(embedding, str):
                    logger.info(f"  First 50 chars: {embedding[:50]}")
            else:
                logger.warning("No embedding generated for text")

        return all_embeddings

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
