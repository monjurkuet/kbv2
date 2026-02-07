#!/usr/bin/env python3
"""Test domain detection debugging"""

import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_base.clients.llm import AsyncLLMClient


async def test_domain_detection():
    llm = AsyncLLMClient()

    # Read the document
    with open("test_data/markdown/advanced_crypto_trading_strategies.md", "r") as f:
        content = f.read()

    # First 2000 chars for testing
    truncated = content[:2000]

    prompt = f"""Analyze the following document and classify it into ONE of these cryptocurrency domains:

Available domains:
- BITCOIN: Bitcoin-specific content (BTC, ETFs, mining, treasuries, halving)
- DEFI: Decentralized finance (Uniswap, Aave, liquidity pools, yield farming)  
- INSTITUTIONAL_CRYPTO: Institutional adoption (BlackRock, corporate treasuries, custody)
- STABLECOINS: Stablecoins (USDC, USDT, DAI, backing mechanisms)
- CRYPTO_REGULATION: Regulation (SEC, laws, compliance, legislation)
- GENERAL: General crypto content (mixed topics, market overview)

Document content:
{truncated}

Respond with ONLY this JSON format (no markdown, no extra text):
{{"domain": "DOMAIN_NAME", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

Use high confidence (>0.7) only if the document clearly focuses on one domain."""

    print("🤖 Testing LLM domain detection...")
    print("=" * 60)

    response = await llm.complete(prompt=prompt, temperature=0.1, max_tokens=200)
    content = response.get("content", "").strip()

    # Clean up
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```json"):
            content = "\n".join(lines[1:])
        else:
            content = "\n".join(lines[1:])
        if content.endswith("```"):
            content = content[:-3]
    content = content.strip()

    print(f"Raw response:\n{content}\n")

    try:
        result = json.loads(content)
        print("✅ Parsed result:")
        print(f"   Domain: {result.get('domain', 'N/A')}")
        print(f"   Confidence: {result.get('confidence', 0)}")
        print(f"   Reasoning: {result.get('reasoning', 'N/A')}")
    except Exception as e:
        print(f"❌ Parse error: {e}")

    await llm.close()


if __name__ == "__main__":
    asyncio.run(test_domain_detection())
