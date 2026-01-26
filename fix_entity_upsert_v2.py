#!/usr/bin/env python3

with open('src/knowledge_base/orchestrator.py', 'r') as f:
    content = f.read()

# Find and replace entity insertion with proper upsert
old_insertion = '''            async with self._vector_store.get_session() as session:
                for entity in all_entities:
                    session.add(entity)

                for edge in all_edges:
                    session.add(edge)

                await session.commit()

                for chunk_id, entity_id in chunk_entity_links:
                    chunk_entity = ChunkEntity(
                        id=uuid4(),
                        chunk_id=chunk_id,
                        entity_id=entity_id,
                    )
                    session.add(chunk_entity)

                await session.commit()'''

new_insertion = '''            async with self._vector_store.get_session() as session:
                from sqlalchemy.dialects.postgresql import insert as pg_insert
                from uuid import uuid4
                
                # Upsert entities to avoid duplicate URI errors
                for entity in all_entities:
                    stmt = pg_insert(Entity).values(
                        id=entity.id,
                        name=entity.name,
                        entity_type=entity.entity_type,
                        description=entity.description,
                        properties=entity.properties,
                        confidence=entity.confidence,
                        uri=entity.uri,
                        source_text=entity.source_text
                    ).on_conflict_do_update(
                        constraint='entities_uri_key',
                        set_={
                            'updated_at': entity.updated_at,
                            'confidence': entity.confidence,
                            'source_text': entity.source_text
                        }
                    )
                    await session.execute(stmt)

                # Insert edges
                for edge in all_edges:
                    try:
                        session.add(edge)
                    except:
                        # Skip duplicate edges
                        pass

                await session.commit()

                # Link chunks to entities (avoid duplicates)
                for chunk_id, entity_id in chunk_entity_links:
                    link_stmt = pg_insert(ChunkEntity).values(
                        id=uuid4(),
                        chunk_id=chunk_id,
                        entity_id=entity_id
                    ).on_conflict_do_nothing()
                    await session.execute(link_stmt)

                await session.commit()'''

if old_insertion in content:
    content = content.replace(old_insertion, new_insertion)
    print("✅ Fixed entity upsert successfully")
else:
    print("❌ Could not find exact insertion code to replace")
    # Try with different whitespace
    import re
    pattern = r'for entity in all_entities:\s+session\.add\(entity\)'
    if re.search(pattern, content):
        print("Found pattern with regex - manual replacement needed")

with open('src/knowledge_base/orchestrator.py', 'w') as f:
    f.write(content)
