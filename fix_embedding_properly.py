#!/usr/bin/env python3

# Read the embedding client
with open('src/knowledge_base/ingestion/v1/embedding_client.py', 'r') as f:
    content = f.read()

# Fix to use the correct Google API format according to docs
old_batch = '''        # Process texts in batches (Google API supports one text per request)
        for text in texts:
            headers = {
                "Content-Type": "application/json",
            }
            if self._api_key:
                headers["x-goog-api-key"] = self._api_key

            response = await client.post(
                f"{self._config.embedding_url}/v1/models/{self._config.embedding_model}:predict",
                headers=headers,
                json={"content": {"parts": [{"text": text}]}},
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract embedding from response
            embedding = data.get("embedding", {}).get("values", [])
            all_embeddings.append(embedding)'''

new_batch = '''        # Process texts - Google embedding API format
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["x-goog-api-key"] = self._api_key
        
        for text in texts:
            response = await client.post(
                f"{self._config.embedding_url}/v1/models/{self._config.embedding_model}:embedContent",
                headers=headers,
                json={"content": {"parts": [{"text": text}]}},
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract embedding from response - handle both formats
            if "embedding" in data:
                embedding = data["embedding"]["values"]
            elif "embeddings" in data.get("prediction", {}):
                # Alternative format
                embedding = data["prediction"]["embeddings"]["values"]
            else:
                embedding = []
            
            all_embeddings.append(embedding)'''

content = content.replace(old_batch, new_batch)

with open('src/knowledge_base/ingestion/v1/embedding_client.py', 'w') as f:
    f.write(content)

print("✅ Fixed to use correct Google embeddings API format")
