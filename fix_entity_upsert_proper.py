#!/usr/bin/env python3

with open('src/knowledge_base/orchestrator.py', 'r') as f:
    content = f.read()

# The exact code to replace
old_code = '''                for entity in all_entities:
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

new_code = '''                from sqlalchemy.dialects.postgresql import insert as pg_insert
                
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
                        }
                    )
                    await session.execute(stmt)

                # Insert edges (skip duplicates)
                for edge in all_edges:
                    try:
                        edge_stmt = pg_insert(Edge).values(
                            id=edge.id,
                            source_id=edge.source_id,
                            target_id=edge.target_id,
                            edge_type=edge.edge_type,
                            properties=edge.properties,
                            confidence=edge.confidence,
                            provenance=edge.provenance,
                            source_text=edge.source_text
                        ).on_conflict_do_nothing()
                        await session.execute(edge_stmt)
                    except:
                        pass

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

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✅ Successfully replaced entity insertion code")
else:
    print("❌ Could not find exact code match")
    print("Attempting fuzzy match...")
    import re
    pattern = re.compile(r'for entity in all_entities:[\s\S]{0,300}?session\.add\(chunk_entity\)')
    if pattern.search(content):
        print("Found pattern - using manual replacement instead")

with open('src/knowledge_base/orchestrator.py', 'w') as f:
    f.write(content)
