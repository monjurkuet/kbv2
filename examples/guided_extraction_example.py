#!/usr/bin/env python3
"""Example: Using Guided Extractor with User Goals"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_base.extraction.guided_extractor import GuidedExtractor
from knowledge_base.clients.llm import AsyncLLMClient


async def automated_extraction_example():
    """Example: Automated extraction (no user goals)."""
    print("=" * 60)
    print("Automated Mode - No User Goals")
    print("=" * 60)

    # Initialize
    llm_client = AsyncLLMClient()
    extractor = GuidedExtractor(llm_client)

    # Example document
    document_text = """
    BlackRock's IBIT ETF has reached $45 billion in AUM since its launch in January 2024. 
    The fund has seen consistent daily inflows, with over $500 million added yesterday.
    Fidelity's FBTC has also performed well, though with lower overall assets.
    Both ETFs track the spot price of Bitcoin and hold actual BTC in custody.
    """

    print("\n📄 Document text:")
    print(document_text[:200] + "...")

    print("\n🤖 Generating prompts for automated extraction...")
    print("   (System will detect domain and use default goals)")

    # Generate prompts (no user goals = automated mode)
    # prompts = await extractor.generate_extraction_prompts(
    #     document_text=document_text,
    #     user_goals=None,  # Automated mode
    # )

    print("\n✅ Would generate prompts for:")
    print("   - Domain detection")
    print("   - Default entity types for detected domain")
    print("   - Default relationship types")

    await llm_client.close()


async def guided_extraction_example():
    """Example: Guided extraction with user goals."""
    print("\n" + "=" * 60)
    print("Guided Mode - With User Goals")
    print("=" * 60)

    llm_client = AsyncLLMClient()
    extractor = GuidedExtractor(llm_client)

    document_text = """
    Strategy (formerly MicroStrategy) holds 471,107 BTC as of Q4 2024.
    Their average cost basis is $30,000 per Bitcoin.
    The company's treasury strategy has been led by Michael Saylor.
    Tesla also holds Bitcoin but has reduced their position.
    """

    # Specific goals from user
    user_goals = [
        "Find all companies holding Bitcoin in their treasury",
        "Extract the amount of BTC each company holds",
        "Identify the leaders of these companies",
    ]

    print("\n📄 Document text:")
    print(document_text[:200] + "...")

    print("\n🎯 User goals:")
    for i, goal in enumerate(user_goals, 1):
        print(f"   {i}. {goal}")

    print("\n🤖 Generating prompts based on user goals...")
    # prompts = await extractor.generate_extraction_prompts(
    #     document_text=document_text,
    #     user_goals=user_goals,
    #     domain="INSTITUTIONAL_CRYPTO",
    # )

    print("\n✅ Would generate prompts for:")
    print("   - DigitalAssetTreasury entities")
    print("   - BTC holding amounts")
    print("   - Company leadership (Person entities)")
    print("   - Relationships: 'leads', 'holds'")

    await llm_client.close()


async def iterative_extraction_example():
    """Example: Iterative extraction (building on previous results)."""
    print("\n" + "=" * 60)
    print("Iterative Mode - Building on Previous Extraction")
    print("=" * 60)

    print("""
This mode is useful when you want to:
1. Do an initial extraction pass
2. Review what was found
3. Do a second pass to find missed information
4. Merge results

Example flow:
  pass1 = await extractor.extract(document, goals=["Find ETFs"])
  pass2 = await extractor.extract(
      document, 
      goals=["Find trading volumes"],
      previous_extraction=pass1  # Don't duplicate ETFs
  )
  final = merge_extractions(pass1, pass2)
""")


async def custom_domain_example():
    """Example: Using custom domain hints."""
    print("\n" + "=" * 60)
    print("Custom Domain Hints")
    print("=" * 60)

    print("""
You can provide domain hints to improve extraction:

 extractor.generate_extraction_prompts(
     document_text=text,
     domain="BITCOIN",  # Force Bitcoin domain
     user_goals=["Find ETF inflows"]
 )

Available domains in template registry:
""")

    from knowledge_base.extraction.template_registry import TemplateRegistry

    registry = TemplateRegistry()

    print("Registered domains:")
    for domain in registry.list_domains():
        print(f"   - {domain}")
        goals = registry.get_default_goals(domain)
        if goals:
            print(f"     Default goals: {len(goals)}")


if __name__ == "__main__":
    # Run examples
    asyncio.run(automated_extraction_example())
    asyncio.run(guided_extraction_example())
    asyncio.run(iterative_extraction_example())
    asyncio.run(custom_domain_example())

    print("\n" + "=" * 60)
    print("Integration with Ingestion Pipeline:")
    print("=" * 60)
    print("""
To integrate guided extraction into the pipeline:

1. Add user_goals parameter to process_document():
   
   document = await orchestrator.process_document(
       file_path="doc.md",
       domain="BITCOIN",
       user_goals=["Find all ETF issuers", "Extract AUM data"]
   )

2. In EntityPipelineService, use GuidedExtractor instead of 
   MultiAgentExtractor when user_goals are provided:
   
   if user_goals:
       extractor = GuidedExtractor(llm_client)
       prompts = await extractor.generate_extraction_prompts(...)
   else:
       extractor = MultiAgentExtractor(...)  # Default

3. Benefits:
   - User-controlled extraction focus
   - Better for specific use cases
   - Can combine with automated mode
""")
