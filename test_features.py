#!/usr/bin/env python3
"""Quick test of domain detection and community summaries"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_base.orchestrator_self_improving import SelfImprovingOrchestrator


async def test():
    """Test ingestion with verbose logging."""
    print("=" * 60)
    print("Testing Domain Detection & Community Summaries")
    print("=" * 60)

    orchestrator = SelfImprovingOrchestrator()
    await orchestrator.initialize()

    try:
        file_path = "test_data/markdown/advanced_crypto_trading_strategies.md"

        print(f"\n📄 Ingesting: {file_path}")
        print("   (No domain specified - will use LLM detection)")

        document = await orchestrator.process_document(
            file_path=file_path,
            document_name="test_doc",
            # domain not specified - should auto-detect
        )

        print(f"\n✅ Document ID: {document.id}")
        print(f"   Domain: {document.domain}")
        print(f"   Status: {document.status}")

        # Check metadata
        if document.doc_metadata:
            print(f"\n📊 Metadata keys: {list(document.doc_metadata.keys())}")

            if "community_summary" in document.doc_metadata:
                summary = document.doc_metadata["community_summary"]
                print(f"\n✅ Community Summary Generated!")
                print(f"   Total communities: {summary.get('total_communities', 0)}")
            else:
                print("\n⚠️ No community summary found")

    finally:
        await orchestrator.close()


if __name__ == "__main__":
    asyncio.run(test())
