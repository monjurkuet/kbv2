#!/usr/bin/env python3
"""Comprehensive domain detection tests"""

import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_base.clients.llm import AsyncLLMClient


TEST_DOCUMENTS = {
    "BITCOIN": """
        Bitcoin ETF approval marks historic moment for cryptocurrency industry.
        BlackRock's IBIT has accumulated over $45 billion in assets under management.
        The Bitcoin halving event reduced miner rewards from 6.25 to 3.125 BTC.
        Mining difficulty has adjusted upward following the halving.
        Strategy (formerly MicroStrategy) holds over 471,000 BTC in their treasury.
        Michael Saylor continues to advocate for Bitcoin as digital gold.
    """,
    "DEFI": """
        Uniswap V3 introduces concentrated liquidity positions for LPs.
        Aave V3 launches on Arbitrum with improved capital efficiency.
        Yield farming strategies on Curve Finance offer 15-25% APY.
        TVL across all DeFi protocols exceeds $100 billion.
        Liquid staking derivatives like Lido's stETH dominate the market.
        Automated Market Makers revolutionized decentralized trading.
    """,
    "TRADING": """
        Ichimoku Cloud analysis reveals strong bullish sentiment on the 4-hour chart.
        Fibonacci retracement levels show support at 0.618.
        Bollinger Bands are squeezing, indicating potential volatility expansion.
        The Stochastic Oscillator shows overbought conditions above 80.
        Order book analysis reveals significant buy walls at $45,000.
        Whale watching shows accumulation patterns in recent sessions.
    """,
    "INSTITUTIONAL_CRYPTO": """
        BlackRock's digital asset custody solution gains regulatory approval.
        Fidelity expands cryptocurrency trading to institutional clients.
        Corporate treasury adoption of Bitcoin accelerates among Fortune 500.
        ETF issuers compete for market share in spot Bitcoin products.
        Custody solutions from Coinbase Prime see increased demand.
        Traditional finance firms establish digital asset departments.
    """,
    "STABLECOINS": """
        USDC reserve attestation shows 100% backing with cash and treasuries.
        The GENIUS Act proposes new regulatory framework for stablecoins.
        Tether's USDT maintains dominance with $100 billion market cap.
        Algorithmic stablecoins face scrutiny following Terra collapse.
        Circle plans IPO amid growing institutional stablecoin demand.
        Cross-border payments increasingly utilize USDC for settlement.
    """,
    "CRYPTO_REGULATION": """
        SEC enforcement actions against unregistered exchanges continue.
        MiCA regulation implementation begins in European Union member states.
        CFTC asserts jurisdiction over cryptocurrency derivatives markets.
        New legislation proposes clearer tax reporting for crypto transactions.
        Regulatory clarity needed for DeFi protocols and DAOs.
        Compliance requirements increase for cryptocurrency exchanges.
    """,
    "GENERAL": """
        The cryptocurrency market has evolved significantly since Bitcoin's creation.
        Blockchain technology enables decentralized applications across industries.
        Digital assets represent a new paradigm in finance and technology.
        Various cryptocurrencies serve different purposes in the ecosystem.
        Market participants include retail investors, institutions, and developers.
        The future of digital assets depends on adoption and regulation.
    """,
    "MIXED": """
        Bitcoin price analysis shows correlation with traditional markets.
        DeFi protocols offer yield opportunities but carry smart contract risks.
        Technical indicators suggest potential breakout above resistance.
        Regulatory developments impact institutional adoption decisions.
        Trading volume has increased across both CEX and DEX platforms.
        The cryptocurrency ecosystem continues to mature and evolve.
    """,
}


async def test_domain(
    client: AsyncLLMClient, expected_domain: str, content: str
) -> dict:
    """Test domain detection for a specific document."""

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
{content[:1500]}

Respond with ONLY this JSON format (no markdown, no extra text):
{{"domain": "DOMAIN_NAME", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

Rules:
- domain must be one of: BITCOIN, DEFI, TRADING, INSTITUTIONAL_CRYPTO, STABLECOINS, CRYPTO_REGULATION, GENERAL
- confidence should reflect how clearly the content matches the domain
- Use GENERAL if content spans multiple domains or is not clearly crypto-related"""

    try:
        response = await client.complete(prompt=prompt, temperature=0.1, max_tokens=200)
        content = response.get("content", "").strip()

        # Clean up
        if "<think>" in content and "</think>" in content:
            think_end = content.find("</think>") + len("</think>")
            content = content[think_end:].strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(
                lines[1:] if not lines[0].startswith("```json") else lines[1:]
            )
            if content.endswith("```"):
                content = content[:-3]
        content = content.strip()

        result = json.loads(content)
        detected = result.get("domain", "GENERAL")
        confidence = result.get("confidence", 0)

        return {
            "expected": expected_domain,
            "detected": detected,
            "confidence": confidence,
            "correct": detected == expected_domain,
            "reasoning": result.get("reasoning", "")[:60],
        }
    except Exception as e:
        return {
            "expected": expected_domain,
            "detected": "ERROR",
            "confidence": 0,
            "correct": False,
            "reasoning": str(e)[:60],
        }


async def run_tests():
    """Run all domain detection tests."""
    print("=" * 80)
    print("COMPREHENSIVE DOMAIN DETECTION TESTS")
    print("=" * 80)

    client = AsyncLLMClient()

    results = []
    for expected_domain, content in TEST_DOCUMENTS.items():
        print(f"\n📝 Testing {expected_domain}...")
        result = await test_domain(client, expected_domain, content)
        results.append(result)

        status = "✅" if result["correct"] else "❌"
        print(
            f"   {status} Expected: {result['expected']:<20} | Detected: {result['detected']:<20} | Confidence: {result['confidence']:.2f}"
        )
        print(f"   Reasoning: {result['reasoning'][:70]}...")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = (correct / total) * 100

    print(f"\n✅ Correct: {correct}/{total} ({accuracy:.1f}%)")
    print(f"❌ Incorrect: {total - correct}/{total}")

    if accuracy >= 85:
        print("\n🎯 Domain detection is working well!")
    elif accuracy >= 70:
        print("\n⚠️  Domain detection needs improvement")
    else:
        print("\n🚨 Domain detection has significant issues")

    # Show incorrect predictions
    incorrect = [r for r in results if not r["correct"]]
    if incorrect:
        print("\n❌ Incorrect Predictions:")
        for r in incorrect:
            print(
                f"   - Expected {r['expected']} but got {r['detected']} (confidence: {r['confidence']:.2f})"
            )

    await client.close()


if __name__ == "__main__":
    asyncio.run(run_tests())
