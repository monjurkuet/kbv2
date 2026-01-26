#!/usr/bin/env python3

# Switch embedding client from Google to Ollama
with open('src/knowledge_base/ingestion/v1/embedding_client.py', 'r') as f:
    content = f.read()

# Replace config
old_config = '''class EmbeddingConfig(BaseSettings):
    """Embedding configuration."""

    model_config = SettingsConfigDict(env_prefix="GOOGLE_", extra="ignore")

    embedding_url: str = "https://generativelanguage.googleapis.com"
    embedding_model: str = "embedding-001"'''

new_config = '''class EmbeddingConfig(BaseSettings):
    """Embedding configuration."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="ignore")

    embedding_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"'''

content = content.replace(old_config, new_config)

# Replace the embed_batch method completely
old_method = '''    async def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Embed multiple texts.

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding vectors.
        """
        client = await self._get_client()
        all_embeddings: list[list[float]] = []

        # Google API processes one text per request
        for text in texts:
            headers = {
                "Content-Type": "application/json",
            }
            if self._api_key:
                headers["x-goog-api-key"] = self._api_key

            response = await client.post(
                f"{self._config.embedding_url}/v1/models/{self._config.embedding_model}:embedContent",
                headers=headers,
                json={"content": {"parts": [{"text": text}]}},
            )
            response.raise_for_status()
            data = response.json()

            # Extract embedding from response - Google API format
            embedding = data.get("embedding", {}).get("values", [])
            if not embedding and "values" in str(data):
                # Fallback for different response format
                try:
                    embedding = data["embedding"]["values"]
                except:
                    embedding = []
                    logger.error(f"Could not parse embedding from response: {data}")
            all_embeddings.append(embedding)

        return all_embeddings'''

new_method = '''    async def embed_batch(
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
            
            all_embeddings.append(embedding)

        return all_embeddings'''

if old_method in content:
    content = content.replace(old_method, new_method)
    print("✅ Replaced embed_batch method completely")
else:
    print("❌ Could not find old method")

# Also remove API key requirement since Ollama doesn't need it
old_init = '''    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize embedding client.

        Args:
            config: Embedding configuration.
        """
        self._config = config or EmbeddingConfig()
        self._client: httpx.AsyncClient | None = None
        self._api_key = os.getenv("GOOGLE_API_KEY", "")
        
        # Don't fail at init if key is missing - Ollama doesn't need it
        if not self._api_key and "google" in self._config.embedding_url:
            logger.warning("GOOGLE_API_KEY not set and using Google endpoint")'''

new_init = '''    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize embedding client.

        Args:
            config: Embedding configuration.
        """
        self._config = config or EmbeddingConfig()
        self._client: httpx.AsyncClient | None = None'''

if old_init in content:
    content = content.replace(old_init, new_init)
    print("✅ Fixed __init__ method")
else:
    print("⚠️ Could not find __init__ pattern")

# Update docstring
content = content.replace('"""Google Embeddings API client."""', '"""Ollama Embeddings API client."""')

with open('src/knowledge_base/ingestion/v1/embedding_client.py', 'w') as f:
    f.write(content)

print("\n" + "="*60)
print("✅ Switched from Google to Ollama embeddings")
print("✅ Model: nomic-embed-text")
print("✅ Endpoint: http://localhost:11434")
print("="*60)
