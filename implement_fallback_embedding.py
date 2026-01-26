#!/usr/bin/env python3

with open('src/knowledge_base/ingestion/v1/embedding_client.py', 'r') as f:
    content = f.read()

# Replace the embedding call with fallback logic
old_usage = '''        # Process texts - Google embedding API format
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
            all_embeddings.append(embedding)'''

new_usage = '''        # Get all API keys from environment
        import os
        all_keys = []
        if self._api_key:
            all_keys.append(self._api_key)
        for i in range(2, 10):
            key = os.getenv(f"GOOGLE_API_KEY_{i}")
            if key:
                all_keys.append(key)
        
        if not all_keys:
            logger.error("No Google API keys found!")
            return [[] for _ in texts]
        
        # Try each key with fallback
        for text in texts:
            embedding = []
            last_error = None
            
            for idx, api_key in enumerate(all_keys):
                try:
                    headers = {
                        "Content-Type": "application/json",
                        "x-goog-api-key": api_key,
                    }
                    
                    response = await client.post(
                        f"{self._config.embedding_url}/v1/models/{self._config.embedding_model}:embedContent",
                        headers=headers,
                        json={"content": {"parts": [{"text": text}]}},
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract embedding from response
                    embedding = data.get("embedding", {}).get("values", [])
                    
                    if embedding:
                        if idx > 0:
                            logger.info(f"✅ Used fallback API key #{idx+1}")
                        break
                    
                except Exception as e:
                    logger.warning(f"❌ API key {idx+1} failed: {str(e)[:80]}")
                    last_error = e
                    continue
            
            if not embedding:
                logger.error(f"❌ All API keys failed for text: {text[:40]}...")
                if last_error:
                    logger.error(f"   Last error: {last_error}")
                embedding = []  # Return empty embedding instead of failing
            
            all_embeddings.append(embedding)'''

content = content.replace(old_usage, new_usage)

with open('src/knowledge_base/ingestion/v1/embedding_client.py', 'w') as f:
    f.write(content)

print("✅ Implemented API key fallback logic")
