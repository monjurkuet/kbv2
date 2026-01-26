#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
import asyncio
import os

# Make sure keys are in environment
keys = {
    "GOOGLE_API_KEY": "AIzaSyA9_aUQ5lMjed-N7PaEYeBwoRA-vgAT1l4",
    "GOOGLE_API_KEY_2": "AIzaSyCdAaZU8TnRrZiJ75pIfNBGwwWQWaCbxOk",
    "GOOGLE_API_KEY_3": "AIzaSyCvX779-2d9j2h00oHBcUZrqxz3lPTCqF4",
}

for k, v in keys.items():
    os.environ[k] = v

from knowledge_base.ingestion.v1.embedding_client import EmbeddingClient

async def test():
    print("Testing embedding client with API keys...")
    client = EmbeddingClient()  # Should not fail
    
    result = await client.embed_text("test")
    print(f"✅ Embedding result: {len(result)} dimensions")
    print(f"✅ Sample: {result[:5]}")
    
    await client.close()

asyncio.run(test())
