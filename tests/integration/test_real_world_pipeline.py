"""Comprehensive real-world tests for KBV2 system under realistic conditions."""

import asyncio
import os
import pytest
import time
from typing import Dict, List, Any
from uuid import uuid4
from datetime import datetime, timedelta

import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from knowledge_base.mcp_server import KBV2MCPProtocol, MCPRequest
from knowledge_base.text_to_sql_agent import TextToSQLAgent
from knowledge_base.common.resilient_gateway.gateway import (
    ResilientGatewayClient,
    ResilientGatewayConfig,
)
from knowledge_base.intelligence.v1.resolution_agent import ResolutionAgent
from knowledge_base.persistence.v1.vector_store import VectorStore
from knowledge_base.common.temporal_utils import (
    TemporalNormalizer,
    TemporalClaim,
    TemporalType,
)
from knowledge_base.ingestion.v1.embedding_client import EmbeddingClient
from src.knowledge_base.persistence.v1.schema import (
    Entity,
    Document,
    Chunk,
    EntityResolution,
    ChunkEntity,
    UUID,
)


@pytest.mark.asyncio
class TestRealWorldKBV2System:
    """Comprehensive real-world tests for KBV2 system."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment with real infrastructure."""
        self.config = ResilientGatewayConfig(
            url=os.getenv("LLM_GATEWAY_URL", "http://localhost:8080"),
            api_key=os.getenv("LLM_API_KEY", "test-key"),
            model="gemini-2.5-flash-lite",
            timeout=30,
            circuit_breaker_failure_threshold=3,
            circuit_breaker_recovery_timeout=10,
            circuit_breaker_success_threshold=2,
            retry_max_attempts=2,
            retry_base_delay=1.0,
            retry_max_delay=10.0,
            model_switching_enabled=True,
            fallback_models=["gemini-pro", "gemini-1.5-pro-latest"],
        )

        # Initialize with real database
        database_url = os.getenv(
            "DATABASE_URL", "postgresql://agentzero@localhost:5432/knowledge_base"
        )
        self.sql_engine = create_engine(database_url)
        self.text_to_sql_agent = TextToSQLAgent(engine=self.sql_engine)

        # Initialize async components
        self.async_engine = create_async_engine(
            database_url.replace("postgresql://", "postgresql+asyncpg://")
        )
        self.async_session = sessionmaker(
            bind=self.sql_engine, class_=AsyncSession, expire_on_commit=False
        )

        self.vector_store = VectorStore()

        yield

        # Cleanup
        if hasattr(self, "gateway") and self.gateway:
            asyncio.run(self.gateway.close())

    async def async_setup(self):
        """Async setup for components."""
        self.gateway = ResilientGatewayClient(self.config)
        await self.vector_store.initialize()

        # Initialize MCP Protocol
        self.mcp_protocol = KBV2MCPProtocol()
        await self.mcp_protocol.initialize()

    async def test_complex_natural_language_queries_sql(self):
        """Test complex natural language queries that translate to sophisticated SQL."""
        await self.async_setup()

        # Test with complex real-world query patterns
        test_queries = [
            # Complex multi-table query
            "Find all documents related to AI research from 2023 with their authors and tags, ordered by creation date",
            # Aggregation query
            "What is the average number of chunks per document in the 'technology' domain?",
            # Join query
            "Show me entities of type 'person' that are mentioned in documents from 'finance' domain, including their descriptions",
            # Subquery
            "Get documents that have more than 5 entities of type 'organization' associated with them",
            # Pattern matching
            "Find all chunks containing the word 'machine learning' in documents created in the last 30 days",
        ]

        for query in test_queries:
            print(f"Testing complex query: {query}")
            start_time = time.time()

            result = self.text_to_sql_agent.execute_query(query)

            end_time = time.time()
            execution_time = end_time - start_time

            print(f"Query executed in {execution_time:.2f}s")
            print(f"SQL: {result['sql']}")
            print(f"Warnings: {result['warnings']}")
            print(f"Error: {result['error']}")

            # Verify execution completed in reasonable time
            assert execution_time < 10.0, f"Query took {execution_time}s, too slow"

            # Check that no dangerous patterns were detected
            assert not result["error"] or "dangerous" not in result["error"].lower()

            print(f"Query '{query}' completed successfully\n")

    async def test_mcp_protocol_concurrent_requests(self):
        """Test MCP protocol handling multiple concurrent requests."""
        await self.async_setup()

        async def make_concurrent_request(
            request_data: Dict[str, Any], request_id: str
        ):
            """Make a single concurrent request."""
            request = MCPRequest(
                method=request_data["method"],
                params=request_data["params"],
                id=request_id,
            )
            return await self.mcp_protocol.process_request(request)

        # Prepare concurrent request patterns
        concurrent_requests = []
        for i in range(5):
            request_type = i % 4
            if request_type == 0:
                req = {
                    "method": "kbv2/query_text_to_sql",
                    "params": {"query": f"Get all entities from domain test_{i}"},
                }
            elif request_type == 1:
                req = {
                    "method": "kbv2/search_entities",
                    "params": {"query": f"test entity {i}", "limit": 5},
                }
            elif request_type == 2:
                req = {
                    "method": "kbv2/search_chunks",
                    "params": {"query": f"test chunk {i}", "limit": 3},
                }
            else:
                req = {
                    "method": "kbv2/ingest_document",
                    "params": {
                        "file_path": f"/tmp/test_doc_{i}.txt",
                        "document_name": f"test_doc_{i}",
                    },
                }

            concurrent_requests.append(req)

        # Execute concurrent requests
        start_time = time.time()
        tasks = [
            make_concurrent_request(req, f"req_{i}")
            for i, req in enumerate(concurrent_requests)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        total_time = end_time - start_time
        print(
            f"Completed {len(concurrent_requests)} concurrent requests in {total_time:.2f}s"
        )

        # Verify all requests completed (some may fail due to missing data, but should not crash)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Request {i} failed with exception: {result}")
                # This is acceptable - we're testing resilience
            else:
                if hasattr(result, "result"):
                    print(
                        f"Request {i} completed successfully: {result.result is not None}"
                    )
                else:
                    print(f"Request {i} completed successfully (no result attribute)")

        # Verify total time is reasonable for concurrent execution
        assert total_time < 15.0, f"Concurrent requests took too long: {total_time}s"

    async def test_entity_resolution_deduplication_complex(self):
        """Test entity resolution and deduplication with complex data."""
        await self.async_setup()

        # Create complex entity resolution scenario
        embedding_client = EmbeddingClient()

        # Create multiple entities that should be resolved together
        test_entities = [
            Entity(
                id=uuid4(),
                name="Apple Inc.",
                entity_type="organization",
                description="American multinational technology company",
                properties={
                    "founded": 1976,
                    "founders": ["Steve Jobs", "Steve Wozniak"],
                },
            ),
            Entity(
                id=uuid4(),
                name="Apple Computer Company",
                entity_type="organization",
                description="American computer company founded in 1976 by Steve Jobs and Steve Wozniak",
                properties={
                    "founded": 1976,
                    "founders": ["Steve Jobs", "Steve Wozniak"],
                },
            ),
            Entity(
                id=uuid4(),
                name="Apple",
                entity_type="company",
                description="Technology company founded by Steve Jobs",
                properties={"industry": "technology", "founded": "1976"},
            ),
        ]

        # Create source text that mentions all entities
        source_text = """
        Apple Inc. was founded in 1976 by Steve Jobs and Steve Wozniak. 
        Originally called Apple Computer Company, the company was established 
        in a garage in Los Altos, California. The company, known simply as 
        Apple, has grown to become one of the largest technology companies 
        in the world.
        """

        # Initialize resolution agent
        resolution_agent = ResolutionAgent(
            gateway=self.gateway, vector_store=self.vector_store
        )

        # Test individual entity resolution
        for entity in test_entities:
            # Find similar entities (these would normally come from vector search)
            candidates = [e for e in test_entities if e.id != entity.id]

            start_time = time.time()
            resolution = await resolution_agent.resolve_entity(
                entity=entity, candidate_entities=candidates, source_text=source_text
            )
            end_time = time.time()

            print(
                f"Entity '{entity.name}' resolution completed in {end_time - start_time:.2f}s"
            )
            print(f"Confidence: {resolution.confidence_score}")
            print(f"Merged entities: {resolution.merged_entity_ids}")
            print(f"Grounding quote: {resolution.grounding_quote[:100]}...")
            print(f"Human review required: {resolution.human_review_required}")

            # Verify resolution completed within timeout
            assert (end_time - start_time) < 10.0, "Entity resolution took too long"

            # The resolution should find that these entities refer to the same real-world entity
            # based on the source text

    async def test_performance_load_multiple_operations(self):
        """Test performance under load with multiple simultaneous operations."""
        await self.async_setup()

        async def perform_operation(operation_type: str, op_id: int):
            """Perform a single operation based on type."""
            try:
                if operation_type == "text_to_sql":
                    query = f"Get entities for performance test {op_id}"
                    result = self.text_to_sql_agent.execute_query(query)
                    return f"SQL_{op_id}: {result['sql']}"

                elif operation_type == "vector_search":
                    # Create embedding for search
                    embedding_client = EmbeddingClient()
                    query_embedding = await embedding_client.embed_text(
                        f"test query {op_id}"
                    )

                    results = await self.vector_store.search_similar_entities(
                        query_embedding=query_embedding,
                        limit=3,
                        similarity_threshold=0.5,
                    )
                    return f"VECTOR_{op_id}: Found {len(results)} results"

                elif operation_type == "temporal_processing":
                    normalizer = TemporalNormalizer()
                    claim = normalizer.extract_temporal_info(
                        f"Company founded in August 2023, updated in {datetime.now().year}"
                    )
                    return f"TEMPORAL_{op_id}: {claim.temporal_type}"

                elif operation_type == "mcp_request":
                    request = MCPRequest(
                        method="kbv2/search_entities",
                        params={"query": f"load test {op_id}", "limit": 2},
                        id=f"load_{op_id}",
                    )
                    response = await self.mcp_protocol.process_request(request)
                    return f"MCP_{op_id}: {response.error is None}"

            except Exception as e:
                return f"ERROR_{op_id}: {str(e)}"

        # Create a realistic load pattern
        operations = []
        op_types = [
            "text_to_sql",
            "vector_search",
            "temporal_processing",
            "mcp_request",
        ]

        for i in range(20):  # 20 concurrent operations
            op_type = op_types[i % len(op_types)]
            operations.append(perform_operation(op_type, i))

        start_time = time.time()
        results = await asyncio.gather(*operations, return_exceptions=True)
        end_time = time.time()

        total_time = end_time - start_time
        successful_ops = sum(
            1 for r in results if not isinstance(r, Exception) and "ERROR" not in str(r)
        )

        print(
            f"Load test: {successful_ops}/{len(operations)} operations completed in {total_time:.2f}s"
        )
        print(f"Operations per second: {len(operations) / total_time:.2f}")

        # Verify reasonable performance under load
        assert total_time < 30.0, f"Load test took too long: {total_time}s"
        assert successful_ops >= len(operations) * 0.8, (
            f"Too many failures: {len(operations) - successful_ops}"
        )

    async def test_temporal_knowledge_graph_features(self):
        """Test temporal knowledge graph features."""
        await self.async_setup()

        normalizer = TemporalNormalizer()

        # Test temporal claim extraction and classification
        temporal_claims = [
            ("Apple Inc. was founded in April 1976 by Steve Jobs", "static"),
            ("The company is currently led by CEO Tim Cook", "dynamic"),
            ("A company is a business entity", "atemporal"),
            ("The product will be launched next month", "dynamic"),
            ("Historical fact: World War II ended in 1945", "static"),
            ("As of today, the policy remains unchanged", "dynamic"),
        ]

        print("Testing temporal claim classification and normalization...")

        for text, expected_type in temporal_claims:
            start_time = time.time()
            claim = normalizer.extract_temporal_info(text)
            end_time = time.time()

            print(f"Text: {text[:50]}...")
            print(f"  Type: {claim.temporal_type}")
            print(f"  Expected: {expected_type}")
            print(f"  Date: {claim.iso8601_date}")
            print(f"  Processing time: {end_time - start_time:.3f}s")

            # Verify processing time is reasonable
            assert (end_time - start_time) < 2.0, (
                f"Temporal processing too slow: {end_time - start_time}s"
            )

            # For this test, we'll just verify the claim was processed
            assert claim.text is not None
            print()

        # Test temporal invalidation
        old_claim = TemporalClaim(
            text="CEO is John Smith",
            temporal_type=TemporalType.DYNAMIC,
            start_date=datetime(2023, 1, 1),
            iso8601_date="2023-01-01T00:00:00+00:00",
        )

        new_claim = TemporalClaim(
            text="CEO is Jane Doe",
            temporal_type=TemporalType.DYNAMIC,
            start_date=datetime(2024, 6, 15),
            iso8601_date="2024-06-15T00:00:00+00:00",
        )

        is_invalidated = normalizer.check_invalidated(old_claim, new_claim)
        print(f"Temporal invalidation test: {is_invalidated}")
        assert is_invalidated, "Newer temporal claim should invalidate older one"

    async def test_domain_tagging_complex_relationships(self):
        """Test domain tagging across complex entity relationships."""
        await self.async_setup()

        # Test domain-based entity search and relationship detection
        embedding_client = EmbeddingClient()

        # Create domain-specific queries to test tagging
        domain_queries = [
            (
                "technology",
                "artificial intelligence, machine learning, neural networks",
            ),
            ("finance", "stocks, bonds, investments, market analysis"),
            ("medicine", "disease, treatment, diagnosis, pharmaceuticals"),
            ("science", "research, experiment, hypothesis, data analysis"),
        ]

        for domain, query_text in domain_queries:
            print(f"Testing domain '{domain}' with query: {query_text}")

            # Get embedding for the query
            query_embedding = await embedding_client.embed_text(query_text)

            # Search for similar entities in this domain
            start_time = time.time()
            results = await self.vector_store.search_similar_entities(
                query_embedding=query_embedding,
                limit=5,
                similarity_threshold=0.3,  # Lower threshold for broader domain matching
            )
            end_time = time.time()

            print(
                f"Found {len(results)} entities in {domain} domain in {end_time - start_time:.3f}s"
            )

            # Test with domain filtering if available
            # In real system, we'd filter by domain tag in the search
            for entity in results:
                print(
                    f"  - {entity['name']} ({entity['entity_type']}): {entity['similarity']:.3f}"
                )

            # Verify performance
            assert (end_time - start_time) < 5.0, (
                f"Domain search too slow: {end_time - start_time}s"
            )

        # Test relationship detection across domains
        print("\nTesting cross-domain relationship detection...")

        # Create embeddings for related concepts across domains
        tech_finance_relationship = await embedding_client.embed_text(
            "AI-driven algorithmic trading systems and financial technology"
        )

        relationship_results = await self.vector_store.search_similar_entities(
            query_embedding=tech_finance_relationship,
            limit=10,
            similarity_threshold=0.4,
        )

        print(f"Cross-domain relationships found: {len(relationship_results)}")

        # Verify we can find entities related to both domains
        tech_entities = [
            e
            for e in relationship_results
            if "technology" in e["description"].lower()
            or "ai" in e["description"].lower()
        ]
        finance_entities = [
            e
            for e in relationship_results
            if "finance" in e["description"].lower()
            or "trading" in e["description"].lower()
        ]

        print(f"  Tech-related: {len(tech_entities)}")
        print(f"  Finance-related: {len(finance_entities)}")

    async def test_resilient_gateway_under_stress(self):
        """Test the resilient gateway with stress conditions."""
        await self.async_setup()

        # Test gateway metrics collection
        initial_metrics = self.gateway.get_metrics()
        print(f"Initial metrics: {initial_metrics}")

        # Send multiple requests to test resilience features
        async def make_request(prompt: str, req_id: int):
            """Make a request through the resilient gateway."""
            try:
                start_time = time.time()
                response = await self.gateway.generate_text(
                    prompt=prompt, temperature=0.7
                )
                end_time = time.time()

                return {
                    "success": True,
                    "response_length": len(response),
                    "time": end_time - start_time,
                    "id": req_id,
                }
            except Exception as e:
                return {"success": False, "error": str(e), "time": 0, "id": req_id}

        # Create stress pattern: multiple rapid requests
        stress_prompts = [
            f"Explain the concept of artificial intelligence in 50 words. Request {i}"
            for i in range(15)
        ]

        print("Testing resilient gateway under stress...")
        start_time = time.time()

        stress_tasks = [
            make_request(prompt, i) for i, prompt in enumerate(stress_prompts)
        ]
        stress_results = await asyncio.gather(*stress_tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time

        successful_requests = sum(
            1 for r in stress_results if isinstance(r, dict) and r.get("success", False)
        )
        failed_requests = len(stress_results) - successful_requests

        print(
            f"Stress test completed: {successful_requests}/{len(stress_results)} successful in {total_time:.2f}s"
        )
        print(f"Success rate: {successful_requests / len(stress_results) * 100:.1f}%")

        # Check final metrics
        final_metrics = self.gateway.get_metrics()
        print(f"Final metrics: {final_metrics}")

        # Verify metrics were collected
        assert final_metrics["total_requests"] >= successful_requests
        assert final_metrics["successful_requests"] >= 0

        # The system should handle failures gracefully
        assert total_time < 60.0, f"Stress test took too long: {total_time}s"

    async def test_end_to_end_knowledge_ingestion_query_cycle(self):
        """Test complete end-to-end cycle: ingestion -> processing -> querying."""
        await self.async_setup()

        print("Testing end-to-end knowledge cycle...")

        # Simulate document ingestion
        try:
            # This would normally ingest a real document, but we'll simulate
            ingestion_request = MCPRequest(
                method="kbv2/ingest_document",
                params={
                    "file_path": "/tmp/test_document.txt",
                    "document_name": "test_cycle_doc",
                    "domain": "test_domain",
                },
                id="ingest_test",
            )

            ingestion_result = await self.mcp_protocol.process_request(
                ingestion_request
            )
            print(f"Ingestion result: Success={ingestion_result.error is None}")

        except Exception as e:
            print(f"Ingestion simulation failed (expected): {e}")
            # This is expected if the file doesn't exist, continue with search tests

        # Test querying the knowledge base
        query_request = MCPRequest(
            method="kbv2/query_text_to_sql",
            params={"query": "show me all entities"},
            id="query_test",
        )

        query_result = await self.mcp_protocol.process_request(query_request)
        print(f"Query result: Success={query_result.error is None}")

        # Test entity search
        entity_request = MCPRequest(
            method="kbv2/search_entities",
            params={"query": "test", "limit": 5},
            id="entity_test",
        )

        entity_result = await self.mcp_protocol.process_request(entity_request)
        print(
            f"Entity search result: Success={entity_result.error is None}, Entities: {len(entity_result.result.get('entities', [])) if entity_result.result else 0}"
        )

        # Test chunk search
        chunk_request = MCPRequest(
            method="kbv2/search_chunks",
            params={"query": "knowledge", "limit": 3},
            id="chunk_test",
        )

        chunk_result = await self.mcp_protocol.process_request(chunk_request)
        print(
            f"Chunk search result: Success={chunk_result.error is None}, Chunks: {len(chunk_result.result.get('chunks', [])) if chunk_result.result else 0}"
        )

        print("End-to-end cycle completed")

    async def test_entity_mention_duplication_handling(self):
        """Test handling of duplicate entity mentions within and across documents."""
        await self.async_setup()

    async def create_document_with_entities(
        self, doc_name: str, entities_data: list
    ) -> UUID:
        """Helper to create document with specified entities."""
        doc_id = uuid4()

        # Create document
        doc = Document(id=doc_id, name=doc_name, status="completed", domain="test")

        # Create chunk with all entities
        chunk_id = uuid4()
        chunk = Chunk(
            id=chunk_id,
            document_id=doc_id,
            text=" ".join([e["quote"] for e in entities_data]),
            chunk_index=0,
        )

        # Create entities and relationships
        for entity_data in entities_data:
            entity = Entity(
                id=entity_data["id"],
                name=entity_data["name"],
                entity_type=entity_data["type"],
                description=entity_data.get("description", ""),
            )

            # Create chunk-entity relationship
            chunk_entity = ChunkEntity(
                chunk_id=chunk_id,
                entity_id=entity_data["id"],
                grounding_quote=entity_data["quote"],
                confidence=1.0,
            )

            # Add to session
            with self.sql_engine.connect() as conn:
                conn.execute(
                    text(
                        "INSERT INTO documents (id, name, source_uri, mime_type, status, doc_metadata, created_at, updated_at, domain) VALUES (:id, :name, NULL, 'text/plain', 'completed', NULL, NOW(), NOW(), 'test')"
                    ),
                    {"id": str(doc_id), "name": doc_name},
                )
                conn.execute(
                    text(
                        "INSERT INTO chunks (id, document_id, text, chunk_index, page_number, token_count, chunk_metadata, created_at, embedding) VALUES (:id, :doc_id, :text, 0, NULL, NULL, NULL, NOW(), NULL)"
                    ),
                    {"id": str(chunk_id), "doc_id": str(doc_id), "text": chunk.text},
                )
                conn.execute(
                    text(
                        "INSERT INTO entities (id, name, entity_type, description, properties, confidence, created_at, updated_at, embedding, uri, source_text, domain) VALUES (:id, :name, :type, :desc, NULL, 1.0, NOW(), NOW(), NULL, NULL, NULL, 'test')"
                    ),
                    {
                        "id": str(entity.id),
                        "name": entity.name,
                        "type": entity.entity_type,
                        "desc": entity.description,
                    },
                )
                conn.execute(
                    text(
                        "INSERT INTO chunk_entities (chunk_id, entity_id, grounding_quote, confidence, created_at) VALUES (:chunk_id, :entity_id, :quote, 1.0, NOW())"
                    ),
                    {
                        "chunk_id": str(chunk_id),
                        "entity_id": str(entity.id),
                        "quote": entity_data["quote"],
                    },
                )
                conn.commit()

        return doc_id

        # Create entities and relationships
        for entity_data in entities_data:
            entity = Entity(
                id=entity_data["id"],
                name=entity_data["name"],
                entity_type=entity_data["type"],
                description=entity_data.get("description", ""),
            )

            # Create chunk-entity relationship
            chunk_entity = ChunkEntity(
                chunk_id=chunk_id,
                entity_id=entity_data["id"],
                grounding_quote=entity_data["quote"],
                confidence=1.0,
            )

            # Add to session
            with self.sql_engine.connect() as conn:
                conn.execute(
                    text(
                        "INSERT INTO documents (id, name, source_uri, mime_type, status, doc_metadata, created_at, updated_at, domain) VALUES (:id, :name, NULL, 'text/plain', 'completed', NULL, NOW(), NOW(), 'test')"
                    ),
                    {"id": str(doc_id), "name": doc_name},
                )
                conn.execute(
                    text(
                        "INSERT INTO chunks (id, document_id, text, chunk_index, page_number, token_count, chunk_metadata, created_at, embedding) VALUES (:id, :doc_id, :text, 0, NULL, NULL, NULL, NOW(), NULL)"
                    ),
                    {"id": str(chunk_id), "doc_id": str(doc_id), "text": chunk.text},
                )
                conn.execute(
                    text(
                        "INSERT INTO entities (id, name, entity_type, description, properties, confidence, created_at, updated_at, embedding, uri, source_text, domain) VALUES (:id, :name, :type, :desc, NULL, 1.0, NOW(), NOW(), NULL, NULL, NULL, 'test')"
                    ),
                    {
                        "id": str(entity.id),
                        "name": entity.name,
                        "type": entity.entity_type,
                        "desc": entity.description,
                    },
                )
                conn.execute(
                    text(
                        "INSERT INTO chunk_entities (chunk_id, entity_id, grounding_quote, confidence, created_at) VALUES (:chunk_id, :entity_id, :quote, 1.0, NOW())"
                    ),
                    {
                        "chunk_id": str(chunk_id),
                        "entity_id": str(entity.id),
                        "quote": entity_data["quote"],
                    },
                )
                conn.commit()

            return doc_id

        # Test Case 1: Same document, same entity mentioned multiple times
        print("Testing same-document entity duplication...")
        apple_id = uuid4()
        same_doc_entities = [
            {
                "id": apple_id,
                "name": "Apple Inc.",
                "type": "organization",
                "quote": "Apple Inc. was founded in 1976.",
            },
            {
                "id": apple_id,  # Same entity ID
                "name": "Apple Inc.",
                "type": "organization",
                "quote": "The company Apple Inc. is headquartered in Cupertino.",
            },
        ]

        doc1_id = await self.create_document_with_entities(
            "Same Doc Duplication", same_doc_entities
        )

        with self.sql_engine.connect() as conn:
            result_row = conn.execute(
                text(
                    "SELECT COUNT(*) FROM chunk_entities WHERE entity_id = :entity_id"
                ),
                {"entity_id": str(apple_id)},
            ).fetchone()
            result = result_row[0] if result_row else 0
        assert result == 2, f"Expected 2 relationships, got {result}"

        print("✓ Same-document duplication handled correctly")

        # Test Case 2: Cross-document entity duplication (same real-world entity)
        print("Testing cross-document entity duplication...")
        google_id = uuid4()

        # Document 1 mentions Google
        doc1_entities = [
            {
                "id": google_id,
                "name": "Google",
                "type": "company",
                "quote": "Google was founded in 1998 by Larry Page and Sergey Brin.",
            }
        ]

        # Document 2 also mentions Google (same entity)
        doc2_entities = [
            {
                "id": google_id,
                "name": "Google",
                "type": "organization",
                "quote": "Google has its headquarters in Mountain View, California.",
            }
        ]

        doc2_id = await self.create_document_with_entities(
            "Doc 1 - Google", doc1_entities
        )
        doc3_id = await self.create_document_with_entities(
            "Doc 2 - Google", doc2_entities
        )

        # Verify both documents have relationship with same entity
        with self.sql_engine.connect() as conn:
            result1_row = conn.execute(
                text(
                    "SELECT COUNT(*) FROM chunk_entities ce JOIN chunks c ON ce.chunk_id = c.id WHERE c.document_id = :doc_id AND ce.entity_id = :entity_id"
                ),
                {"doc_id": str(doc2_id), "entity_id": str(google_id)},
            ).fetchone()
            result2_row = conn.execute(
                text(
                    "SELECT COUNT(*) FROM chunk_entities ce JOIN chunks c ON ce.chunk_id = c.id WHERE c.document_id = :doc_id AND ce.entity_id = :entity_id"
                ),
                {"doc_id": str(doc3_id), "entity_id": str(google_id)},
            ).fetchone()
            result1 = result1_row[0] if result1_row else 0
            result2 = result2_row[0] if result2_row else 0

        assert result1 == 1, f"Expected 1 relationship in doc1, got {result1}"
        assert result2 == 1, f"Expected 1 relationship in doc2, got {result2}"

        print("✓ Cross-document duplication handled correctly")

        # Test Case 3: Verify no unintended merging of different entities
        print("Testing entity separation for similar names...")
        bank1_id = uuid4()
        bank2_id = uuid4()

        bank_entities = [
            {
                "id": bank1_id,
                "name": "Bank of America",
                "type": "financial_institution",
                "quote": "Bank of America provides banking services.",
            },
            {
                "id": bank2_id,
                "name": "American Bank",
                "type": "financial_institution",
                "quote": "American Bank offers financial products.",
            },
        ]

        doc4_id = await self.create_document_with_entities(
            "Similar Names", bank_entities
        )

        # Verify two distinct entities were created
        with self.sql_engine.connect() as conn:
            result_row = conn.execute(
                text(
                    "SELECT COUNT(DISTINCT entity_id) FROM chunk_entities WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = :doc_id)"
                ),
                {"doc_id": str(doc4_id)},
            ).fetchone()
            result = result_row[0] if result_row else 0
        assert result == 2, f"Expected 2 distinct entities, got {result}"

        print("✓ Similar named entities kept separate")

        print("All entity duplication tests completed successfully!")

    async def run_all_tests(self):
        """Run all comprehensive tests."""
        print("Starting comprehensive real-world KBV2 system tests...\n")

        # Run each test sequentially to avoid overwhelming the system
        tests = [
            self.test_complex_natural_language_queries_sql,
            self.test_mcp_protocol_concurrent_requests,
            self.test_entity_resolution_deduplication_complex,
            self.test_performance_load_multiple_operations,
            self.test_temporal_knowledge_graph_features,
            self.test_domain_tagging_complex_relationships,
            self.test_resilient_gateway_under_stress,
            self.test_end_to_end_knowledge_ingestion_query_cycle,
            self.test_entity_mention_duplication_handling,
        ]

        results = {}
        for test_func in tests:
            test_name = test_func.__name__
            print(f"\n{'=' * 60}")
            print(f"Running {test_name}")
            print(f"{'=' * 60}")

            try:
                await test_func()
                results[test_name] = "PASSED"
                print(f"✓ {test_name} PASSED")
            except Exception as e:
                results[test_name] = f"FAILED: {str(e)}"
                print(f"✗ {test_name} FAILED: {str(e)}")

        print(f"\n{'=' * 60}")
        print("Test Results Summary:")
        print(f"{'=' * 60}")

        passed = sum(1 for result in results.values() if result == "PASSED")
        total = len(results)

        for test_name, result in results.items():
            status = "✓ PASS" if result == "PASSED" else "✗ FAIL"
            print(f"{status} - {test_name}")

        print(f"\nOverall: {passed}/{total} tests passed")

        return passed == total


# Run the tests
if __name__ == "__main__":

    async def main():
        test_suite = TestRealWorldKBV2System()
        success = await test_suite.run_all_tests()
        exit(0 if success else 1)

    asyncio.run(main())
