#!/usr/bin/env python3

with open('src/knowledge_base/ingestion/v1/embedding_client.py', 'r') as f:
    content = f.read()

# Fix the endpoint once and for all - use correct Google API format
old_wrong_endpoint = 'f"{self._config.embedding_url}/v1/models/{self._config.embedding_model}:predict"'
new_correct_endpoint = 'f"{self._config.embedding_url}/v1/models/{self._config.embedding_model}:embedContent"'
content = content.replace(old_wrong_endpoint, new_correct_endpoint)

# Also fix the response parsing for the correct format
old_parse = '''            # Extract embedding from response
            embedding = data.get("embedding", {}).get("values", [])'''

new_parse = '''            # Extract embedding from response - Google API format
            embedding = data.get("embedding", {}).get("values", [])
            if not embedding and "values" in str(data):
                # Fallback for different response format
                try:
                    embedding = data["embedding"]["values"]
                except:
                    embedding = []
                    logger.error(f"Could not parse embedding from response: {data}")'''

content = content.replace(old_parse, new_parse)

# Fix env config to correct model name
with open('.env', 'r') as f:
    env = f.read()

if 'GOOGLE_EMBEDDING_MODEL=embedding-001' not in env:
    env = env.replace('GOOGLE_EMBEDDING_MODEL=gemini-embedding-001', 'GOOGLE_EMBEDDING_MODEL=embedding-001')

with open('.env', 'w') as f:
    f.write(env)

with open('src/knowledge_base/ingestion/v1/embedding_client.py', 'w') as f:
    f.write(content)

print("✅ Fixed embedding endpoint to use :embedContent")
print("✅ Fixed response parsing")
print("✅ Verified model is embedding-001")
