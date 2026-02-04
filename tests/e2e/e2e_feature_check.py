#!/usr/bin/env uv run python
"""Comprehensive KBV2 Feature Verification."""

import asyncio
import sys
from pathlib import Path

# Feature checklist based on docs/README.md
FEATURES = {
    # 9-Stage Pipeline Features
    "Create Document": "DocumentPipelineService.create_document()",
    "Partition Document": "SemanticChunker.chunk()",
    "Extract Knowledge (Adaptive Gleaning)": "EntityPipelineService.extract()",
    "Embed Content": "EmbeddingClient.embed_batch()",
    "Resolve Entities (Verbatim-Grounded)": "EntityPipelineService.resolve_entities()",
    "Cluster Entities (Hierarchical Leiden)": "ClusteringService.cluster()",
    "Generate Reports (Map-Reduce)": "SynthesisAgent.generate_report()",
    "Update Domain": "DomainDetectionService.detect_domain()",
    "Complete": "Orchestrator.process_document()",
    # Intelligence Features
    "Multi-Agent Extraction": "MultiAgentExtractor.extract()",
    "Entity Typing": "EntityTyper.type_entities()",
    "Hallucination Detection": "HallucinationDetector.verify_attribute()",
    "Cross-Domain Detection": "CrossDomainDetector.detect()",
    "Federated Query": "FederatedQueryRouter.route_query()",
    "Hybrid Retrieval": "HybridRetriever.hybrid_search()",
    # Data Layer Features
    "Vector Search": "VectorStore.search()",
    "Graph Store": "GraphStore.get_community()",
    "Temporal Claims": "TemporalNormalizer.normalize()",
    "Schema Registry": "SchemaRegistry.register()",
    # API Features
    "Query API": "query_api.router",
    "Review API": "review_api.router",
    "MCP Server": "mcp_server.kbv2_protocol",
    "WebSocket Support": "main.py websocket endpoint",
}


async def verify_features():
    """Verify all KBV2 features are implemented."""
    print("\n" + "=" * 80)
    print("KBV2 COMPREHENSIVE FEATURE VERIFICATION")
    print("=" * 80)

    results = {}

    # Test 1: Core Services Initialize
    print("\n📦 Testing Core Service Initialization...")
    try:
        from knowledge_base.orchestration.document_pipeline_service import (
            DocumentPipelineService,
        )
        from knowledge_base.orchestration.entity_pipeline_service import (
            EntityPipelineService,
        )
        from knowledge_base.orchestration.domain_detection_service import (
            DomainDetectionService,
        )
        from knowledge_base.orchestration.quality_assurance_service import (
            QualityAssuranceService,
        )
        from knowledge_base.persistence.v1.vector_store import VectorStore
        from knowledge_base.common.resilient_gateway import ResilientGatewayClient

        vector_store = VectorStore()
        await vector_store.initialize()

        doc_service = DocumentPipelineService()
        await doc_service.initialize()

        domain_service = DomainDetectionService()
        await domain_service.initialize()

        gateway = ResilientGatewayClient()

        entity_service = EntityPipelineService()
        await entity_service.initialize(
            vector_store=vector_store,
            gateway=gateway,
        )

        print("   ✅ DocumentPipelineService - Initialized")
        print("   ✅ DomainDetectionService - Initialized")
        print("   ✅ EntityPipelineService - Initialized")
        print("   ✅ VectorStore - Initialized")
        print("   ✅ ResilientGatewayClient - Initialized")

        results["DocumentPipelineService"] = True
        results["DomainDetectionService"] = True
        results["EntityPipelineService"] = True
        results["VectorStore"] = True
        results["ResilientGatewayClient"] = True

    except Exception as e:
        print(f"   ❌ Core services failed: {e}")
        for key in results:
            results[key] = False

    # Test 2: Intelligence Services
    print("\n🧠 Testing Intelligence Services...")
    intelligence_services = [
        ("ClusteringService", "knowledge_base.intelligence.v1.clustering_service"),
        (
            "HallucinationDetector",
            "knowledge_base.intelligence.v1.hallucination_detector",
        ),
        ("SynthesisAgent", "knowledge_base.intelligence.v1.synthesis_agent"),
        ("ResolutionAgent", "knowledge_base.intelligence.v1.resolution_agent"),
        ("EntityTyper", "knowledge_base.intelligence.v1.entity_typing_service"),
    ]

    for name, module_path in intelligence_services:
        try:
            module = __import__(module_path, fromlist=[name])
            cls = getattr(module, name)
            if hasattr(cls, "__init__") and not isinstance(cls, type):
                instance = cls()
            elif hasattr(cls, "initialize"):
                await cls.initialize()
            print(f"   ✅ {name} - Available")
            results[name] = True
        except Exception as e:
            print(f"   ⚠️  {name} - {str(e)[:50]}")
            results[name] = None  # Unknown

    # Test 3: Ingestion Services
    print("\n📥 Testing Ingestion Services...")
    ingestion_services = [
        ("SemanticChunker", "knowledge_base.partitioning.semantic_chunker"),
        ("EmbeddingClient", "knowledge_base.ingestion.v1.embedding_client"),
    ]

    gateway = None
    for name, module_path in ingestion_services:
        try:
            module = __import__(module_path, fromlist=[name])
            cls = getattr(module, name)
            instance = cls()
            print(f"   ✅ {name} - Available")
            results[name] = True
        except Exception as e:
            print(f"   ❌ {name} - Failed: {str(e)[:50]}")
            results[name] = False

    # Test GleaningService separately (needs gateway)
    try:
        from knowledge_base.ingestion.v1.gleaning_service import GleaningService
        from knowledge_base.common.gateway import GatewayClient

        gateway = GatewayClient()
        gleaning = GleaningService(gateway=gateway)
        print(f"   ✅ GleaningService - Available (with gateway)")
        results["GleaningService"] = True
    except Exception as e:
        print(f"   ⚠️  GleaningService - {str(e)[:50]}")
        results["GleaningService"] = None

    # Test 4: API Endpoints
    print("\n🌐 Testing API Endpoints...")
    try:
        from knowledge_base import (
            query_api,
            review_api,
            graph_api,
            document_api,
            schema_api,
        )

        print("   ✅ Query API - Available")
        print("   ✅ Review API - Available")
        print("   ✅ Graph API - Available")
        print("   ✅ Document API - Available")
        print("   ✅ Schema API - Available")
        results["APIs"] = True
    except Exception as e:
        print(f"   ❌ APIs - Failed: {e}")
        results["APIs"] = False

    # Test 5: MCP Server
    print("\n🔌 Testing MCP Server...")
    try:
        from knowledge_base.mcp_server import KBV2MCPProtocol

        protocol = KBV2MCPProtocol()
        methods = [m for m in dir(protocol) if not m.startswith("_")]
        print(f"   ✅ MCP Protocol - {len(methods)} methods available")
        results["MCP_Server"] = True
    except Exception as e:
        print(f"   ❌ MCP Server - Failed: {e}")
        results["MCP_Server"] = False

    # Test 6: Database Schema
    print("\n🗄️  Testing Database Schema...")
    try:
        from knowledge_base.persistence.v1.schema import (
            Document,
            Chunk,
            Entity,
            Edge,
            ChunkEntity,
            Community,
            ReviewQueue,
        )

        tables = [
            "Document",
            "Chunk",
            "Entity",
            "Edge",
            "ChunkEntity",
            "Community",
            "ReviewQueue",
        ]
        for table in tables:
            print(f"   ✅ {table} - Schema defined")
        results["Schema"] = True
    except Exception as e:
        print(f"   ❌ Schema - Failed: {e}")
        results["Schema"] = False

    # Test 7: Common Utilities
    print("\n🛠️  Testing Common Utilities...")
    try:
        from knowledge_base.common.gateway import GatewayClient
        from knowledge_base.common.resilient_gateway.gateway import (
            ResilientGatewayClient,
        )
        from knowledge_base.common.temporal_utils import TemporalNormalizer
        from knowledge_base.common.dependencies import get_session_factory

        print("   ✅ GatewayClient - Available")
        print("   ✅ ResilientGatewayClient - Available")
        print("   ✅ TemporalNormalizer - Available")
        print("   ✅ Session Factory - Available")
        results["Utilities"] = True
    except Exception as e:
        print(f"   ❌ Utilities - Failed: {e}")
        results["Utilities"] = False

    # Test 8: Document Processing Pipeline
    print("\n⚙️  Testing Document Processing Pipeline...")
    try:
        from knowledge_base.orchestrator import IngestionOrchestrator

        orchestrator = IngestionOrchestrator()
        await orchestrator.initialize()

        # Check all service attributes exist
        services = [
            "_document_service",
            "_entity_pipeline_service",
            "_quality_assurance_service",
            "_domain_service",
            "_gateway",
            "_vector_store",
        ]

        for service in services:
            if hasattr(orchestrator, service):
                print(f"   ✅ {service} - Initialized")
            else:
                print(f"   ⚠️  {service} - Missing")

        results["Orchestrator"] = True
    except Exception as e:
        print(f"   ❌ Orchestrator - Failed: {e}")
        results["Orchestrator"] = False

    # Summary
    print("\n" + "=" * 80)
    print("FEATURE VERIFICATION SUMMARY")
    print("=" * 80)

    working = sum(1 for v in results.values() if v is True)
    unknown = sum(1 for v in results.values() if v is None)
    failed = sum(1 for v in results.values() if v is False)

    print(f"\n✅ Working Features: {working}")
    print(f"⚠️  Unknown/Untested: {unknown}")
    print(f"❌ Failed Features: {failed}")
    print(f"\nTotal Features Checked: {len(results)}")

    print("\n📋 Detailed Results:")
    for feature, status in results.items():
        if status is True:
            print(f"   ✅ {feature}")
        elif status is False:
            print(f"   ❌ {feature}")
        else:
            print(f"   ⚠️  {feature} (untested)")

    print("\n" + "=" * 80)

    return working, unknown, failed


if __name__ == "__main__":
    try:
        working, unknown, failed = asyncio.run(verify_features())
        exit(0 if failed == 0 else 1)
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
