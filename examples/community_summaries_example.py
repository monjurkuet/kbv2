#!/usr/bin/env python3
"""Example: Using Community Summarizer"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_base.summaries.community_summaries import (
    CommunitySummarizer,
    HierarchyLevel,
)


async def summarize_communities_example():
    """Example: Generate community summaries from entities."""
    print("=" * 60)
    print("Community Summarization Example")
    print("=" * 60)

    summarizer = CommunitySummarizer()

    # Example entities (in real usage, these come from the knowledge graph)
    entities = [
        {"name": "Bitcoin", "type": "Cryptocurrency", "community_id": "macro_1"},
        {"name": "Ethereum", "type": "Cryptocurrency", "community_id": "macro_1"},
        {"name": "BlackRock", "type": "Company", "community_id": "macro_2"},
        {"name": "IBIT", "type": "ETF", "community_id": "macro_2"},
        {"name": "Strategy", "type": "Company", "community_id": "macro_3"},
        {"name": "Michael Saylor", "type": "Person", "community_id": "macro_3"},
    ]

    relationships = [
        {"source": "BlackRock", "target": "IBIT", "type": "issues"},
        {"source": "Strategy", "target": "Bitcoin", "type": "holds"},
        {"source": "Michael Saylor", "target": "Strategy", "type": "leads"},
    ]

    # Generate multi-level summaries
    print("\n🔄 Generating community summaries...\n")

    multi_level = await summarizer.generate_multi_level_summary(
        document_id=str(uuid4()),
        entities=entities,
        relationships=relationships,
    )

    # Display results
    print(f"📊 Document: {multi_level.document_id}\n")

    print(f"🌍 Macro Communities: {len(multi_level.macro_communities)}")
    for comm in multi_level.macro_communities:
        print(f"   - {comm.name}: {comm.summary[:80]}...")

    print(f"\n🏢 Meso Communities: {len(multi_level.meso_communities)}")
    for comm in multi_level.meso_communities:
        print(f"   - {comm.name}: {len(comm.key_entities)} entities")

    print(f"\n🔬 Micro Communities: {len(multi_level.micro_communities)}")

    print("\n📈 Hierarchy Tree:")
    print(multi_level.hierarchy_tree)


async def name_community_example():
    """Example: Use LLM to name a community."""
    print("\n" + "=" * 60)
    print("Community Naming Example")
    print("=" * 60)

    summarizer = CommunitySummarizer()

    # Community data
    entities = ["Bitcoin", "Ethereum", "Solana", "Cardano"]
    relationships = [
        "Bitcoin is the first cryptocurrency",
        "Ethereum enables smart contracts",
        "Solana is a high-performance blockchain",
    ]

    print(f"\n🤔 Naming community with entities: {entities}")

    # This would use LLM to generate a name
    # result = await summarizer.name_community(entities, relationships)
    # print(f"✅ Suggested name: {result.suggested_name}")
    # print(f"   Description: {result.description}")
    # print(f"   Themes: {result.key_themes}")

    print("\n💡 To use this feature, call:")
    print("   summarizer.name_community(entities, relationships)")


async def hierarchy_levels_example():
    """Example: Understanding hierarchy levels."""
    print("\n" + "=" * 60)
    print("Hierarchy Levels Explained")
    print("=" * 60)

    levels = {
        HierarchyLevel.MACRO: "Broad themes (e.g., 'Cryptocurrency Markets')",
        HierarchyLevel.MESO: "Sub-topics (e.g., 'Bitcoin ETFs', 'DeFi Protocols')",
        HierarchyLevel.MICRO: "Specific entities (e.g., 'BlackRock IBIT')",
        HierarchyLevel.NANO: "Atomic entities (e.g., individual transactions)",
    }

    for level, description in levels.items():
        print(f"\n{level.value.upper()}:")
        print(f"   {description}")


if __name__ == "__main__":
    # Run examples
    asyncio.run(summarize_communities_example())
    asyncio.run(name_community_example())
    asyncio.run(hierarchy_levels_example())

    print("\n" + "=" * 60)
    print("Integration with Ingestion Pipeline:")
    print("=" * 60)
    print("""
To integrate community summaries into the ingestion pipeline:

1. After entity extraction and clustering, call:
   summarizer = CommunitySummarizer()
   summary = await summarizer.generate_multi_level_summary(
       document_id=document.id,
       entities=extracted_entities,
       relationships=extracted_relationships,
   )

2. Store summaries in database for later retrieval

3. Use for:
   - Document overview/summary
   - Navigation in knowledge graph
   - Understanding document structure
   - Identifying key themes
""")
