KBV2 DETAILED FIX & IMPROVEMENT REPORT
Executive Summary
This report provides a comprehensive analysis of all issues found in the KBV2 codebase, categorized by severity, with detailed recommendations for fixes and improvements.
---
PART 1: CURRENT STATE OVERVIEW
1.1 What's Working ✅
| Component | Status | Details |
|-----------|--------|---------|
| PostgreSQL Database | ✅ Ready | PostgreSQL 18.1 + pgvector 0.8.1, all 7 tables present |
| Document Ingestion | ✅ Complete | Full 9-stage pipeline in orchestrator.py |
| Semantic Chunking | ✅ Complete | 1536 tokens, 25% overlap, semantic boundaries |
| Multi-Modal Extraction | ✅ Complete | Tables, images, figures in gleaning_service.py |
| Domain Detection | ✅ Complete | Hybrid keyword + LLM approach with 5 domains |
| Hybrid Search | ✅ Complete | BM25 + Vector with weighted fusion |
| Cross-Encoder Reranking | ✅ Complete | Reranking pipeline with RRF fusion |
| Entity Resolution | ✅ Complete | Similarity-based duplicate detection |
| Clustering | ✅ Complete | Leiden algorithm for community detection |
| Graph Store | ✅ Complete | Entities, edges, communities with relationships |
1.2 What's Partially Working ⚠️
| Component | Status | Issue |
|-----------|--------|-------|
| Document Search | ⚠️ Incomplete | Returns empty results, needs VectorStore integration |
| Graph Path Finding | ⚠️ Placeholder | Functions return empty results |
| Graph Export | ⚠️ Empty | Statistics return empty arrays |
| Embedding Model Upgrade | ⚠️ Not Found | No higher-dimension model support (1024-3072) |
| Reranking API Integration | ⚠️ Partial | Pipeline exists but not fully integrated into endpoints |
1.3 What's Broken 🔴
| Component | Severity | Issue |
|-----------|----------|-------|
| Semantic Chunker Overlap | 🔴 CRITICAL | Logic skips sentences instead of including them |
| Duplicate Code | 🔴 CRITICAL | 3 identical blocks in gleaning_service.py |
| Resource Leak | 🔴 CRITICAL | Session closed before use in multi-agent extraction |
| ReviewQueue Relationships | 🟡 MEDIUM | Missing back_populates declarations |
---
PART 2: CRITICAL ISSUES (MUST FIX)
2.1 Duplicate Code in GleaningService
File: src/knowledge_base/ingestion/v1/gleaning_service.py  
Lines: 264-295 (3 identical code blocks)  
Severity: 🔴 CRITICAL
Current State
The same code block for "long-tail distribution handling" is repeated three times:
# Block 1 (lines 264-271)
if len(extracted_entities) > 20:
    long_tail = sorted(extracted_entities, key=lambda x: x.confidence)[-5:]
    # ... handling logic
# Block 2 (lines 274-283) - IDENTICAL
if len(extracted_entities) > 20:
    long_tail = sorted(extracted_entities, key=lambda x: x.confidence)[-5:]
    # ... handling logic
# Block 3 (lines 288-295) - IDENTICAL
if len(extracted_entities) > 20:
    long_tail = sorted(extracted_entities, key=lambda x: x.confidence)[-5:]
    # ... handling logic
Recommended Fix
1. Extract to a method: Create a private method _handle_long_tail_distribution():
def _handle_long_tail_distribution(
    self, 
    extracted_entities: List[ExtractedEntity],
    pass_type: str
) -> List[ExtractedEntity]:
    """Handle long-tail entities with low confidence."""
    if len(extracted_entities) <= 20:
        return extracted_entities
    
    long_tail = sorted(extracted_entities, key=lambda x: x.confidence)[-5:]
    remaining = [e for e in extracted_entities if e not in long_tail]
    
    for entity in long_tail:
        entity.metadata = entity.metadata or {}
        entity.metadata["long_tail"] = True
        entity.metadata["pass_type"] = pass_type
    
    return remaining + long_tail
2. Replace all three blocks with single call:
extracted_entities = self._handle_long_tail_distribution(
    extracted_entities, 
    pass_type
)
3. Delete the duplicate blocks (lines 264-295)
Impact
- Code Quality: Eliminates duplication, improves maintainability
- Functionality: No change (logic is identical)
- Risk: Low - refactoring only
---
2.2 Incorrect Overlap Logic in SemanticChunker
File: src/knowledge_base/partitioning/semantic_chunker.py  
Lines: 349-358  
Severity: 🔴 CRITICAL
Current State
The overlap calculation logic is fundamentally flawed:
# Current logic (INCORRECT)
overlap_tokens -= sent_tokens
if overlap_tokens >= 0:
    continue  # SKIPS the sentence entirely!
This means:
- When overlap budget is exhausted, sentences are SKIPPED
- Instead, sentences should be INCLUDED in the overlap region
- This causes data loss and breaks semantic continuity
Recommended Fix
def _split_into_overlapping_chunks(
    self,
    sentences: List[SentencesWithIndices],
    chunk_size: int,
    overlap_ratio: float
) -> List[Chunk]:
    """Split sentences into overlapping chunks with correct overlap logic."""
    chunks = []
    current_chunk = []
    current_size = 0
    overlap_size = int(chunk_size * overlap_ratio)
    chunk_index = 0
    
    # Track sentences that will be in the overlap region
    overlap_sentences = []
    
    for sentence in sentences:
        sent_size = sentence.end_char - sentence.start_char
        
        if current_size + sent_size <= chunk_size:
            # Fits in current chunk
            current_chunk.append(sentence)
            current_size += sent_size
        else:
            # Finalize current chunk
            chunk_text = self._join_sentences(current_chunk)
            chunk = self._create_chunk(chunk_text, chunk_index, current_chunk)
            chunks.append(chunk)
            chunk_index += 1
            
            # Calculate overlap for NEXT chunk
            overlap_sentences = []
            overlap_size_calc = overlap_size
            
            # Collect sentences for overlap (NOT skip!)
            for overlap_candidate in reversed(current_chunk):
                if overlap_size_calc <= 0:
                    break
                overlap_sentences.insert(0, overlap_candidate)
                overlap_size_calc -= (overlap_candidate.end_char - overlap_candidate.start_char)
            
            # Start new chunk WITH overlap sentences included
            current_chunk = overlap_sentences.copy()
            current_size = sum(
                s.end_char - s.start_char 
                for s in overlap_sentences
            )
            
            # Add current sentence to new chunk
            current_chunk.append(sentence)
            current_size += sent_size
    
    # Handle final chunk
    if current_chunk:
        chunk_text = self._join_sentences(current_chunk)
        chunk = self._create_chunk(chunk_text, chunk_index, current_chunk)
        chunks.append(chunk)
    
    return chunks
Key Changes
1. Include overlap, don't skip: Overlap sentences are added to the NEXT chunk
2. Preserve semantic flow: Chunks maintain context across boundaries
3. Fix data loss: No sentences are discarded when overlap is exhausted
Impact
- Functionality: Fixes data loss in chunk boundaries
- Quality: Improves retrieval accuracy by maintaining semantic continuity
- Risk: Medium - changes chunk generation behavior
---
2.3 Resource Leak in Multi-Agent Extraction
File: src/knowledge_base/orchestrator.py  
Lines: 1306-1318  
Severity: 🔴 CRITICAL
Current State
async def _extract_entities_multi_agent(
    self,
    session: AsyncSession,
    chunks: list[Chunk],
    domain: str,
    document: Document,
) -> tuple[list[EntityCreate], list[EdgeCreate]]:
    # ... code ...
    
    async with self._gateway:  # Session used here
        extraction_manager = EntityExtractionManager(
            community_store=self._community_store,
            graph_store=self._graph_store,
        )
    
    # ERROR: Using extraction_manager AFTER session is closed!
    extraction_result = await extraction_manager.extract(
        entities_to_reExtract=entities_to_reExtract,
        # ... other params
    )
The session/context manager closes before extraction_manager is used, potentially causing database operation failures.
Recommended Fix
async def _extract_entities_multi_agent(
    self,
    session: AsyncSession,
    chunks: list[Chunk],
    domain: str,
    document: Document,
) -> tuple[list[EntityCreate], list[EdgeCreate]]:
    # ... initial setup code ...
    
    # Keep session open during entire extraction
    async with self._gateway:
        extraction_manager = EntityExtractionManager(
            community_store=self._community_store,
            graph_store=self._graph_store,
        )
        
        # Use extraction_manager INSIDE the context
        extraction_result = await extraction_manager.extract(
            entities_to_reExtract=entities_to_reExtract,
            chunks_with_entities=chunks_with_entities,
            document_id=document.id,
            domain=domain,
            hallucination_store=self._hallucination_store,
            cross_domain_detector=self._cross_domain_detector,
            entity_typer=self._entity_typer,
            review_queue=self._review_queue,
            entity_resolver=self._entity_resolver,
            domain_schema_service=self._domain_schema_service,
            content_text=content_text,
        )
    
    # Return results after context closes
    return (
        extraction_result.entities,
        extraction_result.edges,
    )
Alternative Fix (if gateway must close first)
async def _extract_entities_multi_agent(
    self,
    session: AsyncSession,
    chunks: list[Chunk],
    domain: str,
    document: Document,
) -> tuple[list[EntityCreate], list[EdgeCreate]]:
    # ... setup code ...
    
    extraction_manager = EntityExtractionManager(
        community_store=self._community_store,
        graph_store=self._graph_store,
    )
    
    # Execute within gateway context
    async with self._gateway:
        # Perform all database operations within the context
        extraction_result = await extraction_manager.extract(
            entities_to_reExtract=entities_to_reExtract,
            chunks_with_entities=chunks_with_entities,
            document_id=document.id,
            domain=domain,
            hallucination_store=self._hallucination_store,
            cross_domain_detector=self._cross_domain_detector,
            entity_typer=self._entity_typer,
            review_queue=self._review_queue,
            entity_resolver=self._entity_resolver,
            domain_schema_service=self._domain_schema_service,
            content_text=content_text,
        )
    
    return (
        extraction_result.entities,
        extraction_result.edges,
    )
Impact
- Functionality: Prevents database operation failures
- Stability: Ensures session is available during critical operations
- Risk: Critical fix required for production
---
PART 3: MEDIUM SEVERITY ISSUES
3.1 Missing Retry Logic in EmbeddingClient
File: src/knowledge_base/ingestion/v1/embedding_client.py  
Lines: 114-156  
Severity: 🟡 MEDIUM
Current State
HTTP requests to Ollama have no retry mechanism:
- Network issues cause immediate failure
- Rate limiting causes immediate failure
- No exponential backoff
Recommended Fix
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        # ... existing code ...
        self._max_retries = 3
        self._retry_exceptions = (
            httpx.RequestError,
            httpx.TimeoutException,
            httpx.ConnectError,
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(self._retry_exceptions),
    )
    async def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Embed texts with automatic retry on transient failures."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._config.base_url}/api/embeddings",
                json={
                    "model": self._config.model,
                    "texts": texts,
                    "options": {
                        "num_thread": self._config.embedding_threads
                    }
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return [item["embedding"] for item in response.json()["embeddings"]]
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts with retry logic."""
        try:
            return await self._embed_with_retry(texts)
        except Exception as e:
            self._logger.error(f"Embedding failed after retries: {e}")
            # Fallback to empty embeddings or raise
            raise
Impact
- Reliability: Handles transient network issues
- Resilience: Automatic recovery from rate limiting
- Risk: Low - adds fault tolerance
---
3.2 Edge Type Fallback Uses NOTA Incorrectly
File: src/knowledge_base/ingestion/v1/gleaning_service.py  
Lines: 735-742  
Severity: 🟡 MEDIUM
Current State
# Current logic (INCORRECT)
if edge_type_str not in EDGE_TYPE_VALUES:
    edge_type = EdgeType.NOTA  # NOTA means "Not Applicable"!
NOTA (Not Applicable) is semantically wrong for invalid edge types. Should use RELATED_TO as a safe default.
Recommended Fix
# Recommended fix
if edge_type_str not in EDGE_TYPE_VALUES:
    # Use RELATED_TO as default for unknown types
    # NOTA should only be used when edge genuinely doesn't apply
    self._logger.warning(
        f"Unknown edge type '{edge_type_str}', "
        f"defaulting to RELATED_TO"
    )
    edge_type = EdgeType.RELATED_TO
Impact
- Semantic Correctness: Uses appropriate default edge type
- Data Quality: Prevents incorrect NOTA classification
- Risk: Low - improves edge classification
---
3.3 Session Closed Before Vector Search
File: src/knowledge_base/orchestrator.py  
Lines: 702, 720  
Severity: 🟡 MEDIUM
Current State
# Line 702
await session.close()  # Session closed
# Line 720 - trying to use vector search which needs session
similar = await self._vector_store.similarity_search(
    " ".join(entity_names),
    top_k=5,
    filters={"domain": domain}
)
Recommended Fix
async def _resolve_entities(
    self,
    session: AsyncSession,
    entities: list[Entity],
    chunks: list[Chunk],
    domain: str,
) -> tuple[list[Entity], list[Entity], list[Resolution]]:
    """Resolve entities with proper session management."""
    
    try:
        # ... entity processing ...
        
        # Use vector store BEFORE closing session
        if entity_names:
            similar = await self._vector_store.similarity_search(
                " ".join(entity_names),
                top_k=5,
                filters={"domain": domain}
            )
            # Process similar results
        
        # Commit changes
        await session.commit()
        
    except Exception as e:
        await session.rollback()
        raise
    finally:
        # Close session after all operations complete
        await session.close()
    
    return new_entities, merged_entities, resolutions
Impact
- Functionality: Prevents operations on closed session
- Stability: Ensures vector search completes before cleanup
- Risk: Critical for entity resolution
---
3.4 Missing Vector Indexes in Schema
File: src/knowledge_base/persistence/v1/schema.py  
Severity: 🟡 MEDIUM
Current State
Vector indexes are created programmatically in vector_store.py but not defined in the SQLAlchemy schema:
- idx_entity_embedding (lines 118-123)
- idx_chunk_embedding (lines 125-136)
This means Base.metadata.create_all() doesn't create them.
Recommended Fix
Add to schema.py Chunk and Entity models:
class Chunk(Base):
    # ... existing columns ...
    
    __table_args__ = (
        # ... existing indexes ...
        Index(
            "idx_chunk_embedding_ivfflat",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
    )
class Entity(Base):
    # ... existing columns ...
    
    __table_args__ = (
        # ... existing indexes ...
        Index(
            "idx_entity_embedding_ivfflat",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
    )
Alternatively, create a migration script:
# alembic/versions/001_create_vector_indexes.py
def upgrade():
    op.execute("""
        CREATE INDEX idx_chunk_embedding_ivfflat
        ON chunks USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)
    
    op.execute("""
        CREATE INDEX idx_entity_embedding_ivfflat
        ON entities USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)
def downgrade():
    op.execute("DROP INDEX IF EXISTS idx_chunk_embedding_ivfflat")
    op.execute("DROP INDEX IF EXISTS idx_entity_embedding_ivfflat")
Impact
- Performance: Ensures indexes exist after schema creation
- Reliability: Indexes are version-controlled
- Risk: Medium - requires database migration
---
3.5 Missing back_populates on Relationships
File: src/knowledge_base/persistence/v1/schema.py  
Lines: 294, 379-381  
Severity: 🟡 MEDIUM
Current State
# Line 294 - Community.parent lacks back_populates
parent = relationship("Community", remote_side=[id])
# Lines 379-381 - ReviewQueue relationships lack back_populates
entity = relationship("Entity")
edge = relationship("Edge")
document = relationship("Document")
Recommended Fix
class Community(Base):
    # ... existing code ...
    
    parent_id = Column(UUID(as_uuid=True), ForeignKey("communities.id"), nullable=True)
    parent = relationship(
        "Community",
        remote_side=[id],
        back_populates="children"  # Add this
    )
    children = relationship("Community", back_populates="parent")  # Add this
class ReviewQueue(Base):
    # ... existing code ...
    
    entity_id = Column(UUID(as_uuid=True), ForeignKey("entities.id"), nullable=True)
    edge_id = Column(UUID(as_uuid=True), ForeignKey("edges.id"), nullable=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)
    
    entity = relationship("Entity", back_populates="review_items")  # Add to Entity model
    edge = relationship("Edge", back_populates="review_items")      # Add to Edge model
    document = relationship("Document", back_populates="review_items")  # Add to Document model
Impact
- Code Quality: Proper SQLAlchemy relationship definitions
- Maintainability: Enables lazy loading and bidirectional access
- Risk: Low - declarative improvement
---
PART 4: LOW SEVERITY ISSUES
4 in Schema
.1 Duplicate ImportsFile: src/knowledge_base/persistence/v1/schema.py  
Lines: 2-3, 28-29  
Severity: 🟢 LOW
Current State
# Lines 2-3
from datetime import datetime
from enum import Enum
# Lines 28-29 (duplicate)
from datetime import datetime
from enum import Enum
Recommended Fix
Remove duplicate imports (keep only one set at the top of the file).
---
4.2 No File Extension Validation in CLI
File: src/knowledge_base/clients/cli.py  
Severity: 🟢 LOW
Current State
File existence is checked but file type validation is missing. Unsupported types may fail later in the pipeline.
Recommended Fix
Add validation:
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".html"}
def validate_file_type(file_path: str) -> bool:
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return True
---
4.3 Missing Embedding for Gleaning Entities
File: src/knowledge_base/orchestrator.py  
Lines: 540-579  
Severity: 🟢 LOW
Current State
Entities created from gleaning don't get embeddings immediately. Embeddings are only created in _embed_content() which runs after.
Recommended Fix
Consider embedding gleaning entities earlier, or document that this is intentional (to batch embeddings).
---
4.4 No Caching for Domain Detection
File: src/knowledge_base/domain/detection.py  
Severity: 🟢 LOW
Current State
Multiple calls to detect_domain for the same document would redo analysis.
Recommended Fix
Add LRU cache:
from functools import lru_cache
class DomainDetector:
    @lru_cache(maxsize=128)
    async def detect_domain_cached(
        self,
        document_content_hash: str,
        document_text: str,
    ) -> DomainDetectionResult:
        # ... detection logic
---
4.5 No Transaction Retry Logic
Severity: 🟢 LOW
Current State
No automatic retry for failed transactions due to concurrent modifications.
Recommended Fix
from sqlalchemy.exc import OperationalError
from tenacity import retry, stop_after_attempt, retry_if_exception_type
@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(OperationalError)
)
async def save_with_retry(session, model):
    session.add(model)
    await session.commit()
---
PART 5: MISSING FEATURES
5.1 Embedding Model Upgrade (Phase 4.2)
Status: ❌ NOT FOUND  
Priority: 🟡 MEDIUM
Plan Requirement
Support higher-dimension models (1024-3072 dim) for better embedding quality.
Recommended Implementation
Update embedding_client.py config:
class EmbeddingConfig(BaseSettings):
    model: str = "nomic-embed-text"
    dimensions: int = 768  # Current default
    # Add support for higher dimensions:
    # - "nomic-embed-text-v1.5" = 768 dim
    # - "gte-large" = 1024 dim
    # - "e5-large-v2" = 1024 dim
    # - "BAAI/bge-large-en-v1.5" = 1024 dim
    # - "OpenAI/text-embedding-3-large" = 3072 dim
    
    @classmethod
    def for_model(cls, model: str) -> "EmbeddingConfig":
        dimension_map = {
            "nomic-embed-text": 768,
            "gte-large": 1024,
            "e5-large-v2": 1024,
            "bge-large-en-v1.5": 1024,
            "text-embedding-3-large": 3072,
        }
        return cls(dimensions=dimension_map.get(model, 768))
---
5.2 Document Search Implementation
Status: ⚠️ INCOMPLETE  
Priority: 🟡 MEDIUM
Current State
document_api.py lines 610-627 returns empty list - needs VectorStore integration.
Recommended Implementation
@router.get("/documents/search")
async def search_documents(
    query: str = Query(..., description="Search query"),
    domain: Optional[str] = Query(None),
    top_k: int = Query(10, ge=1, le=100),
    search_service: HybridSearchEngine = Depends(get_hybrid_search),
) -> SearchResponse:
    """Search across all indexed documents."""
    results = await search_service.search(
        query=query,
        domain_filter=domain,
        top_k=top_k,
    )
    
    return SearchResponse(
        results=[
            SearchResult(
                document_id=r.document_id,
                chunk_id=r.chunk_id,
                text=r.text,
                score=r.score,
                type="document"
            )
            for r in results
        ],
        total=len(results)
    )
---
5.3 Graph Path Finding Implementation
Status: ⚠️ PLACEHOLDER  
Priority: 🟢 LOW
Current State
graph_api.py lines 575-624 contain placeholder functions returning empty results.
Recommended Implementation
Use igraph for path finding:
async def _find_shortest_paths(
    self,
    graph_id: UUID,
    source_id: UUID,
    target_id: UUID,
) -> List[PathResult]:
    """Find shortest paths using igraph."""
    graph = await self._graph_store.get_graph(graph_id)
    
    # Build igraph
    ig_graph = igraph.Graph()
    for node in graph.nodes:
        ig_graph.add_vertex(name=str(node.id), type=node.type)
    for edge in graph.edges:
        ig_graph.add_edge(
            str(edge.source_id),
            str(edge.target_id),
            weight=1 - edge.confidence  # Higher confidence = shorter path
        )
    
    # Find shortest paths
    paths = ig_graph.get_shortest_paths(
        str(source_id),
        str(target_id),
        weights="weight"
    )
    
    return [
        PathResult(
            nodes=[graph.get_node_by_id(p) for p in path],
            edges=[graph.get_edge_by_nodes(path[i], path[i+1]) for i in range(len(path)-1)],
            length=len(path)
        )
        for path in paths
        if path
    ]
---
PART 6: DATABASE IMPROVEMENTS
6.1 Add Alembic Migrations
Priority: 🟡 MEDIUM  
Current State: No migration system
Recommended Implementation
# Initialize Alembic
alembic init alembic
# alembic/env.py
from knowledge_base.persistence.v1.schema import Base
target_metadata = Base.metadata
# Create initial migration
alembic revision -m "initial_schema"
# Upgrade
alembic upgrade head
---
6.2 Add Unique Constraint to ChunkEntity
Priority: 🟢 LOW  
Current State: No unique constraint on junction table
Recommended Fix
class ChunkEntity(Base):
    # ... existing columns ...
    
    __table_args__ = (
        # ... existing indexes ...
        UniqueConstraint('chunk_id', 'entity_id', name='uq_chunk_entity_pair'),
    )
---
PART 7: SUMMARY OF RECOMMENDED FIXES
By Priority
| Priority | Count | Issues |
|----------|-------|--------|
| 🔴 CRITICAL | 3 | Duplicate code, Overlap logic, Resource leak |
| 🟡 MEDIUM | 6 | Retry logic, Edge type, Session management, Vector indexes, back_populates, Embedding upgrade |
| 🟢 LOW | 5 | Duplicate imports, File validation, Caching, Transaction retry, Path finding |
By Effort
| Effort | Fixes |
|--------|-------|
| 1 hour | Edge type fallback, Duplicate imports |
| 2-4 hours | Duplicate code removal, back_populates |
| 4-8 hours | Overlap logic fix, Retry logic |
| 1-2 days | Session management, Document search |
| 1 week | Embedding upgrade, Graph path finding |
Total Estimated Effort
- Critical Fixes: 1-2 days
- Medium Fixes: 3-5 days
- Low Priority: 2-3 days
- Missing Features: 1-2 weeks
---
PART 8: RECOMMENDED IMPLEMENTATION ORDER
Phase 1: Critical Fixes (Week 1)
1. Day 1-2: Remove duplicate code in gleaning_service.py
2. Day 3-4: Fix semantic chunker overlap logic
3. Day 5: Fix resource leak in orchestrator.py
Phase 2: Stability Improvements (Week 2)
1. Day 1-2: Add retry logic to embedding client
2. Day 3: Fix session management in entity resolution
3. Day 4-5: Add vector indexes to schema/migration
Phase 3: Quality of Life (Week 3)
1. Day 1-2: Implement document search
2. Day 3: Fix edge type fallback
3. Day 4-5: Add back_populates, file validation
Phase 4: Missing Features (Week 4-5)
1. Embedding model upgrade
2. Graph path finding
3. Alembic migrations
4. Transaction retry logic
---