#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

# Load environment
exec(open('load_env.py').read())

import asyncio
import httpx

async def test_api_key():
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_API_KEY_2')
    print(f"Using API key: {api_key[:15]}...")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://generativelanguage.googleapis.com/v1/models/embedding-001:embedContent",
            headers={"x-goog-api-key": api_key},
            json={"content": {"parts": [{"text": "test"}]}},
            timeout=10
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            embedding = data.get("embedding", {}).get("values", [])
            print(f"✅ Embedding received: {len(embedding)} dimensions")
            return True
        else:
            print(f"❌ Error: {response.text[:100]}")
            return False

result = asyncio.run(test_api_key())
print(f"\nResult: {'✅ PASS' if result else '❌ FAIL'}")
sys.exit(0 if result else 1)
