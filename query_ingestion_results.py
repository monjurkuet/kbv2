import asyncio
from knowledge_base.persistence.v1.vector_store import VectorStore
from knowledge_base.persistence.v1.schema import (
    Document,
    Chunk,
    Entity,
    Edge,
    Community,
    ChunkEntity,
)
from sqlalchemy import select, text
from uuid import UUID


async def query_results():
    store = VectorStore()
    await store.initialize()

    doc_id = "5566b111-7302-49a4-af3d-343a344fa26a"

    async with store.get_session() as session:
        doc_result = await session.execute(
            select(Document).where(Document.id == UUID(doc_id))
        )
        doc = doc_result.scalars().first()
        print(f"=== DOCUMENT ===")
        print(f"Name: {doc.name}")
        print(f"Status: {doc.status}")
        print(f"Domain: {doc.domain}")
        print(f"ID: {doc.id}")

        chunks_result = await session.execute(
            select(Chunk).where(Chunk.document_id == UUID(doc_id))
        )
        chunks = chunks_result.scalars().all()
        print(f"\n=== CHUNKS ({len(chunks)}) ===")
        for c in chunks:
            print(f"\n--- Chunk {c.chunk_index} (len={len(c.text)}) ---")
            print(f"{c.text}")

        entities_result = await session.execute(
            text(f"""
            SELECT DISTINCT e.id, e.name, e.entity_type, e.description, e.confidence, e.domain
            FROM entities e
            JOIN chunk_entities ce ON e.id = ce.entity_id
            JOIN chunks c ON ce.chunk_id = c.id
            WHERE c.document_id = '{doc_id}'::uuid
        """)
        )
        entities = entities_result.fetchall()
        print(f"\n=== ENTITIES ({len(entities)}) ===")
        for e in entities:
            print(f"\n  ID: {e[0]}")
            print(f"  Name: {e[1]}")
            print(f"  Type: {e[2]}")
            print(f"  Description: {e[3]}")
            print(f"  Confidence: {e[4]}")
            print(f"  Domain: {e[5]}")

        edges_result = await session.execute(
            text(f"""
            SELECT DISTINCT e.id, e.edge_type, e.source_id, e.target_id, e.confidence, e.provenance, e.source_text
            FROM edges e
            JOIN chunk_entities ce ON e.source_id = ce.entity_id
            JOIN chunks c ON ce.chunk_id = c.id
            WHERE c.document_id = '{doc_id}'::uuid
        """)
        )
        edges = edges_result.fetchall()
        print(f"\n=== EDGES ({len(edges)}) ===")
        for ed in edges:
            print(f"\n  ID: {ed[0]}")
            print(f"  Type: {ed[1]}")
            print(f"  Source ID: {ed[2]}")
            print(f"  Target ID: {ed[3]}")
            print(f"  Confidence: {ed[4]}")
            print(f"  Provenance: {ed[5]}")
            print(f"  Source Text: {ed[6][:100] if ed[6] else 'None'}...")

        comm_result = await session.execute(select(Community))
        communities = comm_result.scalars().all()
        print(f"\n=== COMMUNITIES ({len(communities)}) ===")
        for comm in communities:
            print(f"\n  ID: {comm.id}")
            print(f"  Name: {comm.name}")
            print(
                f"  Summary: {comm.summary[:500] if comm.summary else 'No summary'}..."
            )

    await store.close()


asyncio.run(query_results())
