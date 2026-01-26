#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
import asyncio

from knowledge_base.ingestion.v1.embedding_client import EmbeddingClient

async def test_all_keys():
    client = EmbeddingClient()
    test_text = "Test embedding with fallback"
    
    print("Testing Google API key fallback...")
    print("-" * 50)
    
    embedding = await client.embed_text(test_text)
    
    if embedding:
        print(f"✅ SUCCESS! Embedding length: {len(embedding)}")
        print(f"✅ First 5 values: {embedding[:5]}")
        return True
    else:
        print("❌ All API keys failed")
        return False
    
    await client.close()

result = asyncio.run(test_all_keys())
print(f"\nResult: {'✅ PASS' if result else '❌ FAIL'}")
sys.exit(0 if result else 1)
