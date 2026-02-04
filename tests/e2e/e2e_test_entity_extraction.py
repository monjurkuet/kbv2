#!/usr/bin/env uv run python
"""Test entity extraction specifically."""

import asyncio
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("entity_test")

TEST_CONTENT = """# Tesla Inc Analysis

Tesla, Inc. is an American electric vehicle company founded by Elon Musk.
The company is led by CEO Elon Musk and CFO Vaibhav Taneja.
Tesla competes with Ford, GM, and Rivian.
The company's stock ticker is TSLA on NASDAQ.
"""


async def test_entity_extraction():
    """Test entity extraction pipeline."""
    print("\n" + "=" * 80)
    print("ENTITY EXTRACTION TEST")
    print("=" * 80)

    # Initialize services
    from knowledge_base.orchestration.entity_pipeline_service import (
        EntityPipelineService,
    )
    from knowledge_base.persistence.v1.vector_store import VectorStore
    from knowledge_base.common.resilient_gateway import (
        ResilientGatewayClient,
        ResilientGatewayConfig,
    )
    from knowledge_base.persistence.v1.schema import Document, Chunk
    from uuid import uuid4

    print("\n📦 Initializing services...")

    # Initialize vector store
    vector_store = VectorStore()
    await vector_store.initialize()
    print("   ✅ Vector store initialized")

    # Initialize gateway
    gateway_config = ResilientGatewayConfig(
        continuous_rotation_enabled=True,
    )
    gateway = ResilientGatewayClient(config=gateway_config)
    print("   ✅ Gateway initialized")

    # Initialize entity pipeline
    entity_pipeline = EntityPipelineService()
    await entity_pipeline.initialize(
        vector_store=vector_store,
        gateway=gateway,
    )
    print("   ✅ Entity pipeline initialized")

    # Create test document
    document = Document(
        id=uuid4(),
        name="Test_Tesla_Document",
        source_uri="test://tesla",
        status="pending",
        domain="GENERAL",
    )
    print(f"   ✅ Created test document: {document.id}")

    # Create a single chunk
    chunk = Chunk(
        id=uuid4(),
        document_id=document.id,
        text=TEST_CONTENT,
        chunk_index=0,
    )
    print(f"   ✅ Created chunk")

    # Extract entities
    print("\n🔍 Extracting entities...")
    try:
        entities, edges = await entity_pipeline.extract(
            document=document,
            chunks=[chunk],
            domain="GENERAL",
            use_multi_agent=True,
        )

        print(f"\n✅ Entity extraction completed!")
        print(f"   📊 Entities found: {len(entities)}")
        print(f"   🔗 Edges found: {len(edges)}")

        # Display entities
        for entity in entities:
            name = getattr(entity, "name", "Unknown") or "Unknown"
            e_type = getattr(entity, "entity_type", "Unknown") or "Unknown"
            uri = getattr(entity, "uri", "Unknown") or "Unknown"
            print(f"   🏷️  {name} ({e_type})")
            print(f"       URI: {uri}")

        # Display edges
        for edge in edges:
            src = getattr(edge, "source_id", "Unknown")
            tgt = getattr(edge, "target_id", "Unknown")
            e_type = getattr(edge, "edge_type", "Unknown") or "Unknown"
            print(f"   🔗 {src} --[{e_type}]--> {tgt}")

    except Exception as e:
        print(f"\n❌ Entity extraction failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test clustering if entities exist
    if entities:
        print("\n📊 Testing entity clustering...")
        try:
            await entity_pipeline.resolve_and_cluster(
                document=document,
                entities=entities,
            )
            print("   ✅ Clustering completed")
        except Exception as e:
            print(f"   ⚠️  Clustering failed: {e}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_entity_extraction())
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
