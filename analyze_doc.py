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
from sqlalchemy import select


async def analyze():
    store = VectorStore()
    await store.initialize()

    async with store.get_session() as session:
        doc_result = await session.execute(
            select(Document).where(Document.name.ilike("%Consolidation%"))
        )
        doc = doc_result.scalars().first()
        if not doc:
            print("Document not found!")
            return

        print(f"=== DOCUMENT ===")
        print(f"ID: {doc.id}")
        print(f"Name: {doc.name}")
        print(f"Status: {doc.status}")
        print(f"Domain: {doc.domain}")
        print(f"Created: {doc.created_at}")

        chunks_result = await session.execute(
            select(Chunk).where(Chunk.document_id == doc.id)
        )
        chunks = chunks_result.scalars().all()
        print(f"\n=== CHUNKS ({len(chunks)}) ===")
        for i, c in enumerate(chunks):
            print(f"Chunk {i}: {len(c.text)} chars")

        entities_result = await session.execute(
            select(Entity)
            .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
            .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
            .where(Chunk.document_id == doc.id)
            .distinct()
        )
        entities = entities_result.scalars().all()
        print(f"\n=== ENTITIES ({len(entities)}) ===")
        type_counts = {}
        for e in entities:
            t = e.entity_type or "Unknown"
            type_counts[t] = type_counts.get(t, 0) + 1
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  - {t}: {c}")

        print(f"\nAll Entities:")
        for e in entities[:30]:
            desc = e.description[:100] if e.description else "No desc"
            print(f"  - {e.name} ({e.entity_type})")
            print(f"    {desc}...")

        edges_result = await session.execute(
            select(Edge)
            .join(Entity, Edge.source_id == Entity.id)
            .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
            .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
            .where(Chunk.document_id == doc.id)
            .distinct()
        )
        edges = edges_result.scalars().all()
        print(f"\n=== EDGES ({len(edges)}) ===")
        edge_type_counts = {}
        for ed in edges:
            t = ed.edge_type
            edge_type_counts[t] = edge_type_counts.get(t, 0) + 1
        for t, c in sorted(edge_type_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"  - {t}: {c}")

        print(f"\nSample Edges:")
        for ed in edges[:10]:
            print(f"  - [{ed.edge_type}] {ed.properties}")

        comm_result = await session.execute(select(Community))
        communities = comm_result.scalars().all()
        print(f"\n=== COMMUNITIES ===")
        print(f"Total communities in DB: {len(communities)}")
        for comm in communities[:5]:
            print(
                f"  - {comm.name}: {comm.summary[:100] if comm.summary else 'No summary'}..."
            )

    await store.close()


asyncio.run(analyze())
