#!/usr/bin/env python3

# Fix the embedding endpoint and model
with open('src/knowledge_base/ingestion/v1/embedding_client.py', 'r') as f:
    content = f.read()

# Fix the endpoint URL
old_url = 'f"{self._config.embedding_url}/v1/models/{self._config.embedding_model}:embedContent"'
new_url = 'f"{self._config.embedding_url}/v1/models/{self._config.embedding_model}:predict"'
content = content.replace(old_url, new_url)

# Also update the model if needed (gemini-embedding-001 might not exist)
if 'gemini-embedding-001' in content:
    content = content.replace('gemini-embedding-001', 'embedding-001')

with open('src/knowledge_base/ingestion/v1/embedding_client.py', 'w') as f:
    f.write(content)

# Also fix env config
with open('.env', 'r') as f:
    env_content = f.read()

if 'GOOGLE_EMBEDDING_MODEL=gemini-embedding-001' in env_content:
    env_content = env_content.replace('GOOGLE_EMBEDDING_MODEL=gemini-embedding-001', 'GOOGLE_EMBEDDING_MODEL=embedding-001')

with open('.env', 'w') as f:
    f.write(env_content)

print("✅ Fixed embedding endpoint and model")
print("   - Changed from gemini-embedding-001 to embedding-001")
print("   - Changed endpoint from :embedContent to :predict")
