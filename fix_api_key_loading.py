#!/usr/bin/env python3

with open('src/knowledge_base/ingestion/v1/embedding_client.py', 'r') as f:
    content = f.read()

# Fix the __init__ to not require API key at init time
old_init = '''    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize embedding client.

        Args:
            config: Embedding configuration.
        """
        self._config = config or EmbeddingConfig()
        self._client: httpx.AsyncClient | None = None
        self._api_key = os.getenv("GOOGLE_API_KEY", "")
        
        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")'''

new_init = '''    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize embedding client.

        Args:
            config: Embedding configuration.
        """
        self._config = config or EmbeddingConfig()
        self._client: httpx.AsyncClient | None = None
        self._api_key = os.getenv("GOOGLE_API_KEY", "")
        
        # Don't fail at init if key is missing - fail at call time instead
        if not self._api_key:
            logger.warning("GOOGLE_API_KEY not set - will try fallback keys at call time")'''

content = content.replace(old_init, new_init)

with open('src/knowledge_base/ingestion/v1/embedding_client.py', 'w') as f:
    f.write(content)

print("✅ Fixed API key loading - won't fail at init")
