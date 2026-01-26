#!/usr/bin/env python3

# Update orchestrator to use upsert for entities
with open('src/knowledge_base/orchestrator.py', 'r') as f:
    content = f.read()

# Find the entity insertion section
old_insert = '''                async with self._vector_store.get_session() as session:
                    for entity in all_entities:
                        session.add(entity)

                    for edge in all_edges:
                        session.add(edge)

                    await session.commit()'''

new_insert = '''                async with self._vector_store.get_session() as session:
                    from sqlalchemy.dialects.postgresql import insert as pg_insert
                    
                    # Upsert entities (ignore duplicates based on URI)
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
                        ).on_conflict_do_nothing(  # Or on_conflict_do_update
                            constraint='entities_uri_key'
                        )
                        await session.execute(stmt)

                    # Insert edges
                    for edge in all_edges:
                        session.add(edge)

                    await session.commit()'''

content = content.replace(old_insert, new_insert)

with open('src/knowledge_base/orchestrator.py', 'w') as f:
    f.write(content)

print("✅ Fixed entity insertion to use upsert")
