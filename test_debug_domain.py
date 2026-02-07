#!/usr/bin/env python3
"""Debug domain detection issues"""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_base.clients.llm import AsyncLLMClient


TEST_CASE = """
    Bitcoin ETF approval marks historic moment for cryptocurrency industry.
    BlackRock's IBIT has accumulated over $45 billion in assets under management.
    The Bitcoin halving event reduced miner rewards from 6.25 to 3.125 BTC.
    Mining difficulty has adjusted upward following the halving.
"""


async def debug():
    client = AsyncLLMClient()

    prompt = f"""Analyze the following cryptocurrency document and classify it into the most appropriate domain.

Available domains:
- BITCOIN: Bitcoin-specific content (BTC, ETFs, mining, treasuries, halving)
- DEFI: Decentralized finance (Uniswap, Aave, liquidity pools, yield farming)  
- TRADING: Trading strategies, technical analysis, market microstructure, order books, indicators
- INSTITUTIONAL_CRYPTO: Institutional adoption (BlackRock, corporate treasuries, custody)
- STABLECOINS: Stablecoins (USDC, USDT, DAI, backing mechanisms)
- CRYPTO_REGULATION: Regulation (SEC, laws, compliance, legislation)
- GENERAL: General crypto content (mixed topics, market overview)

Document content:
{TEST_CASE[:1500]}

Respond with ONLY this JSON format (no markdown, no extra text):
{{"domain": "DOMAIN_NAME", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

Rules:
- domain must be one of: BITCOIN, DEFI, TRADING, INSTITUTIONAL_CRYPTO, STABLECOINS, CRYPTO_REGULATION, GENERAL
- confidence should reflect how clearly the content matches the domain
- Use GENERAL if content spans multiple domains or is not clearly crypto-related"""

    print("🤖 Sending request to LLM...")
    print("=" * 60)

    try:
        response = await client.complete(prompt=prompt, temperature=0.1, max_tokens=200)
        content = response.get("content", "")

        print(f"✅ Got response!")
        print(f"Response type: {type(response)}")
        print(
            f"Response keys: {response.keys() if isinstance(response, dict) else 'N/A'}"
        )
        print(f"\nRaw content:")
        print("-" * 60)
        print(content)
        print("-" * 60)
        print(f"\nContent length: {len(content)}")
        print(f"Content empty? {not content.strip()}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()

    await client.close()


if __name__ == "__main__":
    asyncio.run(debug())
