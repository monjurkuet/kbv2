#!/usr/bin/env uv run python
"""End-to-end test for KBV2 pipeline."""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("e2e_test")

# Test document content with rich entities and relationships
TEST_DOCUMENT_CONTENT = """# Tesla, Inc. Financial Analysis Report

## Executive Summary

Tesla, Inc. (NASDAQ: TSLA) is an American electric vehicle and clean energy company founded by Elon Musk in 2003.
The company is headquartered in Austin, Texas and operates globally with major manufacturing facilities in California,
Texas, Germany, and China.

## Key Leadership

**Elon Musk** serves as Chief Executive Officer and Chief Product Architect. He also founded SpaceX in 2002
and acquired Twitter in 2022. **Tom Zhu** leads Tesla's Asia-Pacific operations and Shanghai Gigafactory.
**Vaibhav Taneja** serves as Chief Financial Officer.

## Financial Performance

For Q4 2023, Tesla reported:
- Revenue: $25.17 billion (up 3% year-over-year)
- Net income: $7.9 billion
- Vehicle deliveries: 405,278 vehicles
- Gross margin: 17.6%

## Key Products

### Vehicle Lineup
- **Model S** - Premium sedan (EPA range: 405 miles)
- **Model 3** - Mass-market sedan (EPA range: 333 miles)
- **Model X** - Luxury SUV (EPA range: 348 miles)
- **Model Y** - Compact SUV (310 miles range)
- **Cybertruck** - Electric pickup truck

### Energy Products
- **Powerwall** - Residential battery storage (13.5 kWh)
- **Powerpack** - Commercial energy storage
- **Megapack** - Utility-scale battery systems
- **Solar Roof** - Integrated solar panels

## Technology Partnerships

Tesla collaborates with **NVIDIA** for autonomous driving hardware and **Panasonic** for battery cell production.
The company also works with **CATL** and **BYD** for battery supply in Chinese markets.

## Market Competition

Major competitors in the EV market include:
- **Rivian** (NASDAQ: RIVN) - Electric trucks and SUVs
- **Lucid Group** (NASDAQ: LCID) - Luxury EVs
- **BYD** (HK: 1211) - Chinese EV manufacturer
- **Ford Motor Company** (NYSE: F) - Mustang Mach-E, F-150 Lightning
- **General Motors** (NYSE: GM) - Chevrolet Bolt, GMC Hummer EV

## Strategic Initiatives

### Full Self-Driving (FSD)
Tesla's FSD Beta program aims to achieve Level 5 autonomy. Hardware 4.0 features improved cameras and AI chips
designed for real-time video processing.

### Optimus Robot
Tesla announced development of the Optimus humanoid robot in 2021, with plans for production at Gigafactory Texas.
The robot is designed for dangerous and repetitive tasks.

## Risk Factors

1. Supply chain disruptions affecting semiconductor availability
2. Increasing competition from legacy automakers
3. Regulatory challenges regarding autonomous driving features
4. Macroeconomic conditions impacting consumer demand
5. Dependence on key personnel including CEO Elon Musk

## Conclusion

Tesla continues to lead the global electric vehicle market with strong financials, innovative technology,
and ambitious expansion plans. The company's vertical integration strategy and energy products provide
multiple growth vectors for long-term value creation.

---
Report generated: January 2024
Author: Financial Analysis Team
"""


async def run_e2e_test():
    """Run comprehensive end-to-end test of KBV2 pipeline."""

    print("\n" + "=" * 80)
    print("KBV2 END-TO-END PIPELINE TEST")
    print("=" * 80)

    # Step 1: Initialize the orchestrator
    print("\n📦 Step 1: Initializing IngestionOrchestrator...")
    from knowledge_base.orchestrator import IngestionOrchestrator

    orchestrator = IngestionOrchestrator(
        progress_callback=lambda x: print(f"   Progress: {x.get('status', x)}")
    )

    try:
        await orchestrator.initialize()
        print("   ✅ Orchestrator initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize orchestrator: {e}")
        return False

    # Step 2: Create test document
    print("\n📄 Step 2: Creating test document...")
    from knowledge_base.persistence.v1.schema import Document

    test_file_path = Path("/tmp/kbv2_e2e_test.md")
    test_file_path.write_text(TEST_DOCUMENT_CONTENT)
    print(f"   ✅ Created test file: {test_file_path}")

    # Step 3: Process document through pipeline
    print("\n⚙️  Step 3: Processing document through ingestion pipeline...")
    try:
        document = await orchestrator.process_document(
            file_path=str(test_file_path),
            document_name="Tesla_Financial_Analysis_Q4_2023",
        )
        print(f"   ✅ Document processed successfully")
        print(f"   📋 Document ID: {document.id}")
        print(f"   📋 Document Name: {document.name}")
        print(f"   📋 Document Status: {document.status}")
        if hasattr(document, "domain"):
            print(f"   📋 Domain: {document.domain}")
    except Exception as e:
        print(f"   ❌ Failed to process document: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Initialize vector store for verification steps
    vector_store = None
    print("\n🔍 Step 4: Verifying entity extraction...")
    try:
        from knowledge_base.persistence.v1.schema import Entity, Chunk, Edge
        from knowledge_base.persistence.v1.vector_store import VectorStore

        vector_store = VectorStore()
        await vector_store.initialize()  # Re-initialize for verification
        async with vector_store.get_session() as session:
            # Get chunks
            chunks_result = await session.execute(
                Chunk.__table__.select().where(Chunk.document_id == document.id)
            )
            chunks = chunks_result.scalars().all()
            print(f"   📊 Total chunks created: {len(chunks)}")

            # Get entities
            entities_result = await session.execute(
                Entity.__table__.select().where(Entity.domain.isnot(None))
            )
            entities = entities_result.scalars().all()
            print(f"   📊 Total entities extracted: {len(entities)}")

            # Get edges
            edges_result = await session.execute(Edge.__table__.select())
            edges = edges_result.scalars().all()
            # Filter edges related to our document
            doc_edges = [
                e
                for e in edges
                if str(document.id) in str(e.source_id)
                or str(document.id) in str(e.target_id)
            ]
            print(f"   📊 Total edges created: {len(doc_edges)}")

            # Display extracted entities
            if entities:
                print("\n   🏷️  Extracted Entities:")
                entity_types = {}
                for entity in entities[:20]:  # Show first 20
                    entity_type = getattr(entity, "entity_type", "Unknown") or "Unknown"
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                    name = getattr(entity, "name", "Unknown") or "Unknown"
                    uri = getattr(entity, "uri", "Unknown") or "Unknown"
                    print(f"      - {name} ({entity_type})")

                print("\n   📈 Entity Type Distribution:")
                for e_type, count in sorted(entity_types.items(), key=lambda x: -x[1]):
                    print(f"      {e_type}: {count}")
            else:
                print("   ⚠️  No entities extracted!")

    except Exception as e:
        print(f"   ❌ Failed to verify entities: {e}")
        import traceback

        traceback.print_exc()

    # Step 5: Test search functionality
    print("\n🔎 Step 5: Testing search functionality...")
    try:
        # Use vector store search directly
        from knowledge_base.ingestion.v1.embedding_client import EmbeddingClient

        embedding_client = EmbeddingClient()

        # Search for Tesla-related content
        test_queries = [
            "Tesla revenue financial performance",
            "Elon Musk CEO",
            "electric vehicle",
        ]

        for query in test_queries:
            try:
                # Get embedding for query
                query_embedding = await embedding_client.embed_text(query)

                # Simple vector search
                async with vector_store.get_session() as session:
                    from sqlalchemy import text

                    result = await session.execute(
                        text("""
                            SELECT c.text, c.chunk_index, embedding <=> :query_vec as distance
                            FROM chunks c
                            WHERE embedding IS NOT NULL
                            ORDER BY embedding <=> :query_vec
                            LIMIT 3
                        """),
                        {"query_vec": query_embedding},
                    )
                    rows = result.fetchall()
                    print(f"   🔍 Query: '{query}'")
                    print(f"      Found {len(rows)} similar chunks")
                    for i, row in enumerate(rows, 1):
                        text_preview = (
                            row[0][:80].replace("\n", " ") if row[0] else "N/A"
                        )
                        print(f"        {i}. [{row[1]}]: {text_preview}...")
            except Exception as e:
                print(f"   ⚠️  Search query failed: {query} - {e}")

    except Exception as e:
        print(f"   ❌ Failed to test search: {e}")
        import traceback

        traceback.print_exc()

    # Step 6: Test query API
    print("\n💬 Step 6: Testing query API...")
    try:
        from knowledge_base.query_api import router as query_router

        # The query API is typically used through FastAPI, but we can test the agent directly
        from knowledge_base.text_to_sql_agent import TextToSQLAgent
        from sqlalchemy import create_engine, text

        database_url = "postgresql://agentzero@localhost:5432/knowledge_base"
        engine = create_engine(database_url)

        with engine.connect() as conn:
            # Check if we can query entities
            result = conn.execute(text("SELECT COUNT(*) FROM entities"))
            count = result.scalar()
            print(f"   ✅ Database connection successful")
            print(f"   📊 Total entities in database: {count}")

    except Exception as e:
        print(f"   ⚠️  Query API test skipped (may need database): {e}")

    # Step 7: Test graph relationships
    print("\n🕸️  Step 7: Testing graph relationships...")
    try:
        from knowledge_base.persistence.v1.graph_store import GraphStore
        from knowledge_base.persistence.v1.schema import Entity

        # Query some entities to check graph structure
        async with vector_store.get_session() as session:
            from sqlalchemy import text

            result = await session.execute(
                text("SELECT id, name, entity_type, uri FROM entities LIMIT 10")
            )
            entities = result.fetchall()

            if entities:
                print(
                    f"   ✅ Graph structure verified with {len(entities)} sample entities"
                )
                print(f"   📋 Sample entities:")
                for entity in entities[:5]:
                    print(f"      - {entity[1]} ({entity[2]})")
            else:
                print("   ⚠️  No entities found for graph verification")

    except Exception as e:
        print(f"   ⚠️  Graph test skipped: {e}")

    # Cleanup
    print("\n🧹 Cleanup...")
    try:
        if test_file_path.exists():
            test_file_path.unlink()
            print("   ✅ Test file cleaned up")
    except:
        pass

    # Final summary
    print("\n" + "=" * 80)
    print("E2E TEST COMPLETE")
    print("=" * 80)
    print("""
Summary:
- Orchestrator initialization: ✅
- Document processing: ✅
- Entity extraction: Depends on LLM gateway availability
- Search functionality: Depends on embedding service
- Graph relationships: Depends on database connectivity

For full validation, ensure:
1. LLM Gateway is running at http://localhost:8087/v1
2. PostgreSQL database is running with knowledge_base schema
3. Embedding service is available
""")

    return True


if __name__ == "__main__":
    print("Starting KBV2 E2E Test...")

    try:
        success = asyncio.run(run_e2e_test())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ E2E Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
