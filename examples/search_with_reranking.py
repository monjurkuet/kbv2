#!/usr/bin/env python3
"""Example: Using the Reranking Pipeline for Search"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_base.storage.hybrid_search import HybridSearchEngine
from knowledge_base.reranking.cross_encoder import CrossEncoderReranker
from knowledge_base.reranking.rrf_fuser import ReciprocalRankFuser
from knowledge_base.reranking.reranking_pipeline import RerankingPipeline


async def search_with_reranking(query: str):
    """Search with reranking pipeline."""
    print(f"🔍 Searching: {query}\n")

    # Initialize components
    hybrid_search = HybridSearchEngine()  # Vector + BM25
    cross_encoder = CrossEncoderReranker()  # Cross-encoder reranker
    rrf_fuser = ReciprocalRankFuser()  # For multi-query fusion

    # Create pipeline
    pipeline = RerankingPipeline(
        hybrid_search=hybrid_search,
        cross_encoder=cross_encoder,
        rr_fuser=rrf_fuser,
    )

    # Search with reranking
    results = await pipeline.search(
        query=query,
        initial_top_k=50,  # Get 50 candidates from hybrid search
        final_top_k=10,  # Return top 10 after reranking
    )

    print(f"✅ Found {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.reranked_score:.3f}")
        print(f"   Text: {result.text[:100]}...")
        print(f"   Cross-encoder: {result.cross_encoder_score:.3f}")
        print()


async def search_with_explanation(query: str):
    """Search with detailed explanation of ranking."""
    from knowledge_base.storage.hybrid_search import HybridSearchEngine
    from knowledge_base.reranking.cross_encoder import CrossEncoderReranker

    print(f"🔍 Searching with explanation: {query}\n")

    hybrid = HybridSearchEngine()
    reranker = CrossEncoderReranker()
    pipeline = RerankingPipeline(hybrid, reranker)

    results = await pipeline.search_with_explanation(
        query=query,
        initial_top_k=20,
        final_top_k=5,
    )

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.id}")
        print(f"   Score: {result.reranked_score:.3f}")
        print(f"   Explanation: {result.explanation}")
        print(f"   Factors: {result.rank_factors}")
        print()


async def multi_query_search(queries: list[str]):
    """Search with multiple queries and fuse results."""
    from knowledge_base.storage.hybrid_search import HybridSearchEngine
    from knowledge_base.reranking.cross_encoder import CrossEncoderReranker
    from knowledge_base.reranking.rrf_fuser import ReciprocalRankFuser

    print(f"🔍 Multi-query search: {queries}\n")

    hybrid = HybridSearchEngine()
    reranker = CrossEncoderReranker()
    rrf = ReciprocalRankFuser()
    pipeline = RerankingPipeline(hybrid, reranker, rrf)

    # Search each query and fuse results
    results = await pipeline.multi_query_search(
        queries=queries,
        initial_top_k=30,
        final_top_k=10,
    )

    print(f"✅ Fused {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.reranked_score:.3f}")
        print(f"   Text: {result.text[:100]}...")
        print()


if __name__ == "__main__":
    # Example 1: Basic reranking
    print("=" * 60)
    print("Example 1: Basic Search with Reranking")
    print("=" * 60)
    asyncio.run(search_with_reranking("Bitcoin ETF approval"))

    # Example 2: Search with explanation
    print("\n" + "=" * 60)
    print("Example 2: Search with Explanation")
    print("=" * 60)
    asyncio.run(search_with_explanation("DeFi liquidity pools"))

    # Example 3: Multi-query search
    print("\n" + "=" * 60)
    print("Example 3: Multi-Query Search with RRF")
    print("=" * 60)
    asyncio.run(
        multi_query_search(
            [
                "Bitcoin price prediction",
                "BTC market analysis",
                "cryptocurrency trading",
            ]
        )
    )
