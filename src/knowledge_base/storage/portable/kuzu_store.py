"""Kuzu graph database for knowledge graph storage and querying.

Kuzu provides:
- Embedded graph database (no server required)
- Cypher query language support
- Native support for entity-relationship modeling
- Single directory persistence for portability

This module wraps Kuzu for use in the portable storage layer.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from knowledge_base.storage.portable.config import KuzuConfig

logger = logging.getLogger(__name__)

# Lazy import for Kuzu
_kuzu = None


def _get_kuzu():
    """Lazy import of kuzu."""
    global _kuzu
    if _kuzu is None:
        import kuzu
        _kuzu = kuzu
    return _kuzu


class Entity(BaseModel):
    """Entity/Node in the knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    entity_type: str = "Entity"
    description: Optional[str] = None
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    domain: Optional[str] = None
    source_text: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Edge(BaseModel):
    """Relationship/Edge in the knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    target_id: str
    relation_type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    source_text: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Community(BaseModel):
    """Community cluster in the knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    level: int = 0
    summary: Optional[str] = None
    entity_count: int = 0
    parent_id: Optional[str] = None


class GraphQueryResult(BaseModel):
    """Result from a graph query."""

    records: list[dict[str, Any]]
    query_time_ms: float


class KuzuGraphStore:
    """Kuzu-based graph database for knowledge graph storage.

    This class provides a fully portable graph database using Kuzu with:
    - Cypher query language support
    - Entity and relationship management
    - Community detection support
    - Single-directory persistence

    Example:
        >>> store = KuzuGraphStore()
        >>> await store.initialize()
        >>> entity_id = await store.add_entity(Entity(name="Bitcoin", entity_type="Cryptocurrency"))
        >>> results = await store.query("MATCH (e:Entity) RETURN e")
    """

    def __init__(self, config: Optional[KuzuConfig] = None) -> None:
        """Initialize Kuzu graph store.

        Args:
            config: Kuzu configuration. Uses defaults if not provided.
        """
        self._config = config or KuzuConfig()
        self._db: Optional[Any] = None
        self._conn: Optional[Any] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Kuzu database and create schema."""
        # Ensure directory exists
        self._config.db_path.parent.mkdir(parents=True, exist_ok=True)

        def _init():
            kuzu = _get_kuzu()

            # Create database
            self._db = kuzu.Database(self._config.db_path_str)
            self._conn = kuzu.Connection(self._db)

            # Create schema
            self._create_schema()

        await asyncio.get_event_loop().run_in_executor(None, _init)

        self._initialized = True
        logger.info(f"Kuzu graph database initialized at {self._config.db_path}")

    def _create_schema(self) -> None:
        """Create the graph schema with nodes and relationships."""
        # Node tables

        # Document node
        self._conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Document (
                id STRING PRIMARY KEY,
                name STRING,
                source_uri STRING,
                domain STRING,
                status STRING,
                created_at TIMESTAMP,
                metadata STRING
            )
        """)

        # Chunk node
        self._conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Chunk (
                id STRING PRIMARY KEY,
                document_id STRING,
                text STRING,
                chunk_index INT64,
                token_count INT64,
                created_at TIMESTAMP
            )
        """)

        # Entity node
        self._conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Entity (
                id STRING PRIMARY KEY,
                name STRING,
                type STRING,
                description STRING,
                domain STRING,
                confidence DOUBLE,
                source_text STRING,
                properties STRING,
                created_at TIMESTAMP
            )
        """)

        # Community node
        self._conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Community (
                id STRING PRIMARY KEY,
                name STRING,
                level INT64,
                summary STRING,
                entity_count INT64,
                parent_id STRING
            )
        """)

        # Relationship tables

        # Chunk is part of Document
        self._conn.execute("""
            CREATE REL TABLE IF NOT EXISTS PART_OF (
                FROM Chunk TO Document
            )
        """)

        # Chunk mentions Entity
        self._conn.execute("""
            CREATE REL TABLE IF NOT EXISTS MENTIONS (
                FROM Chunk TO Entity,
                confidence DOUBLE,
                quote STRING
            )
        """)

        # Entity relates to Entity
        self._conn.execute("""
            CREATE REL TABLE IF NOT EXISTS RELATES_TO (
                FROM Entity TO Entity,
                relation_type STRING,
                confidence DOUBLE,
                source_text STRING
            )
        """)

        # Entity is member of Community
        self._conn.execute("""
            CREATE REL TABLE IF NOT EXISTS MEMBER_OF (
                FROM Entity TO Community
            )
        """)

        # Entity is subclass of Entity
        self._conn.execute("""
            CREATE REL TABLE IF NOT EXISTS SUBCLASS_OF (
                FROM Entity TO Entity
            )
        """)

        logger.debug("Graph schema created successfully")

    async def close(self) -> None:
        """Close database connection."""
        self._conn = None
        self._db = None
        self._initialized = False

    # ==================== Entity Operations ====================

    async def add_entity(self, entity: Entity) -> str:
        """Add an entity to the graph.

        Args:
            entity: Entity to add.

        Returns:
            Entity ID.
        """
        def _add():
            self._conn.execute("""
                CREATE (e:Entity {
                    id: $id,
                    name: $name,
                    type: $type,
                    description: $description,
                    domain: $domain,
                    confidence: $confidence,
                    source_text: $source_text,
                    properties: $properties,
                    created_at: $created_at
                })
            """, {
                "id": entity.id,
                "name": entity.name,
                "type": entity.entity_type,
                "description": entity.description or "",
                "domain": entity.domain or "",
                "confidence": entity.confidence,
                "source_text": entity.source_text or "",
                "properties": json.dumps(entity.properties),
                "created_at": entity.created_at.isoformat(),
            })
            return entity.id

        return await asyncio.get_event_loop().run_in_executor(None, _add)

    async def add_entities_batch(self, entities: list[Entity]) -> list[str]:
        """Add multiple entities in a batch.

        Args:
            entities: List of entities to add.

        Returns:
            List of entity IDs.
        """
        def _add():
            entity_ids = []
            for entity in entities:
                self._conn.execute("""
                    CREATE (e:Entity {
                        id: $id,
                        name: $name,
                        type: $type,
                        description: $description,
                        domain: $domain,
                        confidence: $confidence,
                        source_text: $source_text,
                        properties: $properties,
                        created_at: $created_at
                    })
                """, {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.entity_type,
                    "description": entity.description or "",
                    "domain": entity.domain or "",
                    "confidence": entity.confidence,
                    "source_text": entity.source_text or "",
                    "properties": json.dumps(entity.properties),
                    "created_at": entity.created_at.isoformat(),
                })
                entity_ids.append(entity.id)
            return entity_ids

        return await asyncio.get_event_loop().run_in_executor(None, _add)

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID.

        Args:
            entity_id: Entity ID.

        Returns:
            Entity if found, None otherwise.
        """
        def _get():
            result = self._conn.execute("""
                MATCH (e:Entity {id: $id})
                RETURN e.id, e.name, e.type, e.description, e.domain,
                       e.confidence, e.source_text, e.properties, e.created_at
            """, {"id": entity_id})

            if result.has_next():
                row = result.get_next()
                return Entity(
                    id=row[0],
                    name=row[1],
                    entity_type=row[2],
                    description=row[3] if row[3] else None,
                    domain=row[4] if row[4] else None,
                    confidence=row[5],
                    source_text=row[6] if row[6] else None,
                    properties=json.loads(row[7]) if row[7] else {},
                    created_at=datetime.fromisoformat(row[8]) if row[8] else datetime.utcnow(),
                )
            return None

        return await asyncio.get_event_loop().run_in_executor(None, _get)

    async def search_entities(
        self,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        domain: Optional[str] = None,
        limit: int = 100,
    ) -> list[Entity]:
        """Search entities by attributes.

        Args:
            name: Filter by name (partial match).
            entity_type: Filter by type.
            domain: Filter by domain.
            limit: Maximum results.

        Returns:
            List of matching entities.
        """
        def _search():
            query = "MATCH (e:Entity) WHERE 1=1"
            params: dict[str, Any] = {}

            if name:
                query += " AND e.name CONTAINS $name"
                params["name"] = name
            if entity_type:
                query += " AND e.type = $type"
                params["type"] = entity_type
            if domain:
                query += " AND e.domain = $domain"
                params["domain"] = domain

            query += f" RETURN e.id, e.name, e.type, e.description, e.domain, e.confidence, e.source_text, e.properties, e.created_at LIMIT {limit}"

            result = self._conn.execute(query, params)
            entities = []

            while result.has_next():
                row = result.get_next()
                entities.append(Entity(
                    id=row[0],
                    name=row[1],
                    entity_type=row[2],
                    description=row[3] if row[3] else None,
                    domain=row[4] if row[4] else None,
                    confidence=row[5],
                    source_text=row[6] if row[6] else None,
                    properties=json.loads(row[7]) if row[7] else {},
                    created_at=datetime.fromisoformat(row[8]) if row[8] else datetime.utcnow(),
                ))

            return entities

        return await asyncio.get_event_loop().run_in_executor(None, _search)

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity by ID.

        Args:
            entity_id: Entity ID.

        Returns:
            True if deleted, False if not found.
        """
        def _delete():
            result = self._conn.execute("""
                MATCH (e:Entity {id: $id})
                DELETE e
                RETURN count(e)
            """, {"id": entity_id})

            return result.has_next() and result.get_next()[0] > 0

        return await asyncio.get_event_loop().run_in_executor(None, _delete)

    # ==================== Edge/Relationship Operations ====================

    async def add_edge(self, edge: Edge) -> str:
        """Add a relationship between entities.

        Args:
            edge: Edge to add.

        Returns:
            Edge ID (stored in properties).
        """
        def _add():
            properties = {**edge.properties, "_edge_id": edge.id}

            self._conn.execute("""
                MATCH (source:Entity {id: $source_id})
                MATCH (target:Entity {id: $target_id})
                CREATE (source)-[r:RELATES_TO {
                    relation_type: $relation_type,
                    confidence: $confidence,
                    source_text: $source_text,
                    properties: $properties
                }]->(target)
            """, {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation_type": edge.relation_type,
                "confidence": edge.confidence,
                "source_text": edge.source_text or "",
                "properties": json.dumps(properties),
            })
            return edge.id

        return await asyncio.get_event_loop().run_in_executor(None, _add)

    async def add_edge_batch(self, edges: list[Edge]) -> list[str]:
        """Add multiple edges in a batch.

        Args:
            edges: List of edges to add.

        Returns:
            List of edge IDs.
        """
        def _add():
            edge_ids = []
            for edge in edges:
                properties = {**edge.properties, "_edge_id": edge.id}
                self._conn.execute("""
                    MATCH (source:Entity {id: $source_id})
                    MATCH (target:Entity {id: $target_id})
                    CREATE (source)-[r:RELATES_TO {
                        relation_type: $relation_type,
                        confidence: $confidence,
                        source_text: $source_text,
                        properties: $properties
                    }]->(target)
                """, {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation_type": edge.relation_type,
                    "confidence": edge.confidence,
                    "source_text": edge.source_text or "",
                    "properties": json.dumps(properties),
                })
                edge_ids.append(edge.id)
            return edge_ids

        return await asyncio.get_event_loop().run_in_executor(None, _add)

    async def get_entity_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        relation_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get relationships for an entity.

        Args:
            entity_id: Entity ID.
            direction: 'outgoing', 'incoming', or 'both'.
            relation_type: Filter by relation type.
            limit: Maximum results.

        Returns:
            List of relationships with connected entities.
        """
        def _get():
            relationships = []

            if direction in ["outgoing", "both"]:
                query = """
                    MATCH (e:Entity {id: $id})-[r:RELATES_TO]->(target:Entity)
                """
                if relation_type:
                    query += " WHERE r.relation_type = $relation_type"
                query += """
                    RETURN e.id as source_id, target.id as target_id, target.name as target_name,
                           r.relation_type, r.confidence, r.source_text, 'outgoing' as direction
                    LIMIT $limit
                """

                result = self._conn.execute(query, {
                    "id": entity_id,
                    "relation_type": relation_type,
                    "limit": limit,
                })

                while result.has_next():
                    row = result.get_next()
                    relationships.append({
                        "source_id": row[0],
                        "target_id": row[1],
                        "target_name": row[2],
                        "relation_type": row[3],
                        "confidence": row[4],
                        "source_text": row[5],
                        "direction": row[6],
                    })

            if direction in ["incoming", "both"]:
                query = """
                    MATCH (source:Entity)-[r:RELATES_TO]->(e:Entity {id: $id})
                """
                if relation_type:
                    query += " WHERE r.relation_type = $relation_type"
                query += """
                    RETURN source.id as source_id, source.name as source_name, e.id as target_id,
                           r.relation_type, r.confidence, r.source_text, 'incoming' as direction
                    LIMIT $limit
                """

                result = self._conn.execute(query, {
                    "id": entity_id,
                    "relation_type": relation_type,
                    "limit": limit,
                })

                while result.has_next():
                    row = result.get_next()
                    relationships.append({
                        "source_id": row[0],
                        "source_name": row[1],
                        "target_id": row[2],
                        "relation_type": row[3],
                        "confidence": row[4],
                        "source_text": row[5],
                        "direction": row[6],
                    })

            return relationships[:limit]

        return await asyncio.get_event_loop().run_in_executor(None, _get)

    # ==================== Chunk Operations ====================

    async def add_chunk_mention(
        self,
        chunk_id: str,
        entity_id: str,
        confidence: float = 1.0,
        quote: Optional[str] = None,
    ) -> None:
        """Create a MENTIONS relationship from a chunk to an entity.

        Args:
            chunk_id: Chunk ID.
            entity_id: Entity ID.
            confidence: Confidence score.
            quote: Grounding quote.
        """
        def _add():
            self._conn.execute("""
                MATCH (c:Chunk {id: $chunk_id})
                MATCH (e:Entity {id: $entity_id})
                CREATE (c)-[r:MENTIONS {confidence: $confidence, quote: $quote}]->(e)
            """, {
                "chunk_id": chunk_id,
                "entity_id": entity_id,
                "confidence": confidence,
                "quote": quote or "",
            })

        await asyncio.get_event_loop().run_in_executor(None, _add)

    async def get_chunk_entities(self, chunk_id: str) -> list[Entity]:
        """Get all entities mentioned in a chunk.

        Args:
            chunk_id: Chunk ID.

        Returns:
            List of entities mentioned in the chunk.
        """
        def _get():
            result = self._conn.execute("""
                MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS]->(e:Entity)
                RETURN e.id, e.name, e.type, e.description, e.domain,
                       e.confidence, e.source_text, e.properties, e.created_at
            """, {"chunk_id": chunk_id})

            entities = []
            while result.has_next():
                row = result.get_next()
                entities.append(Entity(
                    id=row[0],
                    name=row[1],
                    entity_type=row[2],
                    description=row[3] if row[3] else None,
                    domain=row[4] if row[4] else None,
                    confidence=row[5],
                    source_text=row[6] if row[6] else None,
                    properties=json.loads(row[7]) if row[7] else {},
                    created_at=datetime.fromisoformat(row[8]) if row[8] else datetime.utcnow(),
                ))

            return entities

        return await asyncio.get_event_loop().run_in_executor(None, _get)

    # ==================== Community Operations ====================

    async def add_community(self, community: Community) -> str:
        """Add a community to the graph.

        Args:
            community: Community to add.

        Returns:
            Community ID.
        """
        def _add():
            self._conn.execute("""
                CREATE (c:Community {
                    id: $id,
                    name: $name,
                    level: $level,
                    summary: $summary,
                    entity_count: $entity_count,
                    parent_id: $parent_id
                })
            """, {
                "id": community.id,
                "name": community.name,
                "level": community.level,
                "summary": community.summary or "",
                "entity_count": community.entity_count,
                "parent_id": community.parent_id or "",
            })
            return community.id

        return await asyncio.get_event_loop().run_in_executor(None, _add)

    async def assign_entity_to_community(self, entity_id: str, community_id: str) -> None:
        """Assign an entity to a community.

        Args:
            entity_id: Entity ID.
            community_id: Community ID.
        """
        def _assign():
            self._conn.execute("""
                MATCH (e:Entity {id: $entity_id})
                MATCH (c:Community {id: $community_id})
                CREATE (e)-[:MEMBER_OF]->(c)
            """, {"entity_id": entity_id, "community_id": community_id})

        await asyncio.get_event_loop().run_in_executor(None, _assign)

    async def get_community_entities(self, community_id: str) -> list[Entity]:
        """Get all entities in a community.

        Args:
            community_id: Community ID.

        Returns:
            List of entities in the community.
        """
        def _get():
            result = self._conn.execute("""
                MATCH (e:Entity)-[:MEMBER_OF]->(c:Community {id: $community_id})
                RETURN e.id, e.name, e.type, e.description, e.domain,
                       e.confidence, e.source_text, e.properties, e.created_at
            """, {"community_id": community_id})

            entities = []
            while result.has_next():
                row = result.get_next()
                entities.append(Entity(
                    id=row[0],
                    name=row[1],
                    entity_type=row[2],
                    description=row[3] if row[3] else None,
                    domain=row[4] if row[4] else None,
                    confidence=row[5],
                    source_text=row[6] if row[6] else None,
                    properties=json.loads(row[7]) if row[7] else {},
                    created_at=datetime.fromisoformat(row[8]) if row[8] else datetime.utcnow(),
                ))

            return entities

        return await asyncio.get_event_loop().run_in_executor(None, _get)

    # ==================== Graph Traversal ====================

    async def traverse(
        self,
        start_entity_id: str,
        max_depth: int = 2,
        relation_types: Optional[list[str]] = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Traverse the graph starting from an entity.

        Args:
            start_entity_id: Starting entity ID.
            max_depth: Maximum traversal depth.
            relation_types: Filter by relation types.
            limit: Maximum results.

        Returns:
            Dictionary with nodes and edges.
        """
        def _traverse():
            # Get starting entity
            nodes = []
            edges = []
            visited = set()

            # BFS traversal
            queue = [(start_entity_id, 0)]

            while queue and len(nodes) < limit:
                current_id, depth = queue.pop(0)

                if current_id in visited or depth > max_depth:
                    continue

                visited.add(current_id)

                # Get entity
                entity = self._conn.execute("""
                    MATCH (e:Entity {id: $id})
                    RETURN e.id, e.name, e.type
                """, {"id": current_id})

                if entity.has_next():
                    row = entity.get_next()
                    nodes.append({
                        "id": row[0],
                        "name": row[1],
                        "type": row[2],
                    })

                if depth < max_depth:
                    # Get connected entities
                    rel_filter = ""
                    if relation_types:
                        rel_filter = f" AND r.relation_type IN {relation_types}"

                    result = self._conn.execute(f"""
                        MATCH (e:Entity {{id: $id}})-[r:RELATES_TO]-(connected:Entity)
                        WHERE 1=1 {rel_filter}
                        RETURN connected.id, connected.name, connected.type,
                               r.relation_type, r.confidence,
                               CASE WHEN startNode(r) = e THEN 'outgoing' ELSE 'incoming' END as direction
                    """, {"id": current_id})

                    while result.has_next():
                        row = result.get_next()
                        connected_id = row[0]

                        edges.append({
                            "source": current_id if row[5] == "outgoing" else connected_id,
                            "target": connected_id if row[5] == "outgoing" else current_id,
                            "relation_type": row[3],
                            "confidence": row[4],
                        })

                        if connected_id not in visited:
                            queue.append((connected_id, depth + 1))

            return {"nodes": nodes, "edges": edges}

        return await asyncio.get_event_loop().run_in_executor(None, _traverse)

    # ==================== Raw Query ====================

    async def query(self, cypher_query: str, params: Optional[dict[str, Any]] = None) -> GraphQueryResult:
        """Execute a raw Cypher query.

        Args:
            cypher_query: Cypher query string.
            params: Query parameters.

        Returns:
            Query result with records and timing.
        """
        import time

        def _execute():
            start_time = time.time()

            result = self._conn.execute(cypher_query, params or {})

            records = []
            columns = []

            # Get column names
            if result.has_next():
                first_row = result.get_next()
                columns = [f"column_{i}" for i in range(len(first_row))]
                records.append(dict(zip(columns, first_row)))

            while result.has_next():
                row = result.get_next()
                records.append(dict(zip(columns, row)))

            query_time = (time.time() - start_time) * 1000

            return GraphQueryResult(records=records, query_time_ms=query_time)

        return await asyncio.get_event_loop().run_in_executor(None, _execute)

    # ==================== Statistics ====================

    async def get_stats(self) -> dict[str, Any]:
        """Get graph statistics.

        Returns:
            Dictionary with statistics.
        """
        def _get():
            entity_count = self._conn.execute("MATCH (e:Entity) RETURN count(e)").get_next()[0]
            chunk_count = self._conn.execute("MATCH (c:Chunk) RETURN count(c)").get_next()[0]
            document_count = self._conn.execute("MATCH (d:Document) RETURN count(d)").get_next()[0]
            community_count = self._conn.execute("MATCH (c:Community) RETURN count(c)").get_next()[0]
            rel_count = self._conn.execute("MATCH ()-[r:RELATES_TO]->() RETURN count(r)").get_next()[0]
            mention_count = self._conn.execute("MATCH ()-[r:MENTIONS]->() RETURN count(r)").get_next()[0]

            # Get directory size
            dir_size = 0
            if self._config.db_path.exists():
                for f in self._config.db_path.rglob("*"):
                    if f.is_file():
                        dir_size += f.stat().st_size

            return {
                "entities": entity_count,
                "chunks": chunk_count,
                "documents": document_count,
                "communities": community_count,
                "relationships": rel_count,
                "mentions": mention_count,
                "storage_size_mb": dir_size / (1024 * 1024),
            }

        return await asyncio.get_event_loop().run_in_executor(None, _get)

    # ==================== Context Managers ====================

    async def __aenter__(self) -> "KuzuGraphStore":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
