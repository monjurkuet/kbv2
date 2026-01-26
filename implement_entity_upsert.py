#!/usr/bin/env python3

with open('src/knowledge_base/orchestrator.py', 'r') as f:
    content = f.read()

# Find the entity insertion section and replace with upsert
old_code = '''                async with self._vector_store.get_session() as session:
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

new_code = '''                async with self._vector_store.get_session() as session:
                    from sqlalchemy.dialects.postgresql import insert as pg_insert
                    from uuid import uuid4
                    
                    # Upsert entities to avoid duplicates
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

                    # Insert edges (can have duplicates, but we'll handle it)
                    for edge in all_edges:
                        try:
                            session.add(edge)
                        except:
                            pass  # Skip if edge already exists

                    await session.commit()

                    # Link chunks to entities
                    for chunk_id, entity_id in chunk_entity_links:
                        link_stmt = pg_insert(ChunkEntity).values(
                            id=uuid4(),
                            chunk_id=chunk_id,
                            entity_id=entity_id
                        ).on_conflict_do_nothing()
                        await session.execute(link_stmt)

                    await session.commit()'''

content = content.replace(old_code, new_code)

with open('src/knowledge_base/orchestrator.py', 'w') as f:
    f.write(content)

print("✅ Implemented entity upsert logic")
