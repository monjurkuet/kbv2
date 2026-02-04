import asyncio
from knowledge_base.persistence.v1.vector_store import VectorStore
from knowledge_base.persistence.v1.schema import (
    Document,
    Chunk,
    Entity,
    Edge,
    Community,
)
from sqlalchemy import text, select, create_engine


def analyze_sync():
    engine = create_engine("postgresql://agentzero@localhost:5432/knowledge_base")
    doc_id = "a077fc3e-bba2-48c0-ae52-b32b4fd920e3"

    with engine.connect() as conn:
        doc_result = conn.execute(
            text("SELECT id, name, status, domain FROM documents WHERE id = :doc_id"),
            {"doc_id": doc_id},
        )
        doc = doc_result.fetchone()
        if not doc:
            print("Document not found!")
            return

        print("=" * 60)
        print("DOCUMENT")
        print("=" * 60)
        print(f"ID: {doc[0]}")
        print(f"Name: {doc[1]}")
        print(f"Status: {doc[2]}")
        print(f"Domain: {doc[3]}")

        chunks_result = conn.execute(
            text("SELECT id, text FROM chunks WHERE document_id = :doc_id"),
            {"doc_id": doc_id},
        )
        chunks = chunks_result.fetchall()
        print(f"\n{'=' * 60}")
        print(f"CHUNKS: {len(chunks)}")
        print("=" * 60)
        for i, c in enumerate(chunks):
            print(f"  Chunk {i}: {len(c[1])} chars")

        entities_result = conn.execute(
            text("""
                SELECT DISTINCT e.id, e.name, e.entity_type, e.description, 
                       e.confidence, e.created_at, e.updated_at,
                       e.embedding, e.uri, e.source_text, e.domain, e.community_id
                FROM entities e
                JOIN chunk_entities ce ON e.id = ce.entity_id
                JOIN chunks c ON ce.chunk_id = c.id
                WHERE c.document_id = :doc_id
            """),
            {"doc_id": doc_id},
        )
        entities = entities_result.fetchall()
        print(f"\n{'=' * 60}")
        print(f"ENTITIES: {len(entities)}")
        print("=" * 60)

        type_counts = {}
        for e in entities:
            t = e[2] or "Unknown"
            type_counts[t] = type_counts.get(t, 0) + 1
        print("By Type:")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")

        print("\nTop 20 Entities:")
        for e in entities[:20]:
            desc = (
                (e[3][:80] + "...") if e[3] and len(e[3]) > 80 else (e[3] or "No desc")
            )
            print(f"  - {e[1]} ({e[2]})")
            print(f"    {desc}")
            print(f"    Has embedding: {e[7] is not None}")

        edges_result = conn.execute(
            text("""
                SELECT DISTINCT edges.id, edges.edge_type, 
                       edges.confidence, edges.created_at,
                       edges.source_id, edges.target_id
                FROM edges
                JOIN entities source ON edges.source_id = source.id
                JOIN chunk_entities ce ON source.id = ce.entity_id
                JOIN chunks c ON ce.chunk_id = c.id
                WHERE c.document_id = :doc_id
            """),
            {"doc_id": doc_id},
        )
        edges = edges_result.fetchall()
        print(f"\n{'=' * 60}")
        print(f"EDGES: {len(edges)}")
        print("=" * 60)

        edge_type_counts = {}
        for ed in edges:
            t = ed[1]
            edge_type_counts[t] = edge_type_counts.get(t, 0) + 1
        print("Top 15 Edge Types:")
        for t, c in sorted(edge_type_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"  {t}: {c}")

        print("\nSample Edges:")
        for ed in edges[:10]:
            print(f"  - [{ed[1]}]")
            print(f"    Source: {ed[4]}, Target: {ed[5]}")

        print(f"\n{'=' * 60}")
        print("COMMUNITIES")
        print("=" * 60)
        doc_entity_ids = {e[0] for e in entities}
        comm_result = conn.execute(
            text("SELECT id, name, level, summary FROM communities")
        )
        communities = comm_result.fetchall()
        relevant_communities = []
        for comm in communities:
            comm_entities_result = conn.execute(
                text("SELECT id FROM entities WHERE community_id = :comm_id"),
                {"comm_id": comm[0]},
            )
            comm_entities = comm_entities_result.fetchall()
            if any(e[0] in doc_entity_ids for e in comm_entities):
                relevant_communities.append((comm, comm_entities))

        print(f"Relevant communities: {len(relevant_communities)}")
        for comm, comm_entities in relevant_communities:
            print(f"\n  Community: {comm[1]}")
            print(f"  Level: {comm[2]}")
            print(f"  Entities: {len(comm_entities)}")
            print(f"  Summary: {comm[3][:300] if comm[3] else 'No summary'}...")

    engine.dispose()


analyze_sync()
