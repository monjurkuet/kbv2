#!/usr/bin/env python3

with open('src/knowledge_base/ingestion/v1/embedding_client.py', 'r') as f:
    content = f.read()

# Remove the requirement for API key at init time
old_lines = '''        self._config = config or EmbeddingConfig()
        self._client: httpx.AsyncClient | None = None
        self._api_key = os.getenv("GOOGLE_API_KEY", "")

        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")'''

new_lines = '''        self._config = config or EmbeddingConfig()
        self._client: httpx.AsyncClient | None = None
        self._api_key = os.getenv("GOOGLE_API_KEY", "")'''

if old_lines in content:
    content = content.replace(old_lines, new_lines)
    print("✅ Removed API key requirement at initialization")
else:
    print("❌ Pattern not found")

with open('src/knowledge_base/ingestion/v1/embedding_client.py', 'w') as f:
    f.write(content)
