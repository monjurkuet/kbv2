import asyncio
from knowledge_base.persistence.v1.vector_store import VectorStore
from knowledge_base.persistence.v1.schema import (
    Document,
    Chunk,
    Entity,
    Edge,
    Community,
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

        entities_result = await session.execute(select(Entity))
        entities = entities_result.scalars().all()
        print(f"\n=== TOTAL ENTITIES IN DB ({len(entities)}) ===")

        edges_result = await session.execute(select(Edge))
        edges = edges_result.scalars().all()
        print(f"\n=== TOTAL EDGES IN DB ({len(edges)}) ===")

        comm_result = await session.execute(select(Community))
        communities = comm_result.scalars().all()
        print(f"\n=== COMMUNITIES ===")
        print(f"Total communities in DB: {len(communities)}")

    await store.close()


asyncio.run(analyze())
