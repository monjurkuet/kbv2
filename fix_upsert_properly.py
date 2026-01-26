#!/usr/bin/env python3

# Fix the upsert properly - use exclude parameter
with open('src/knowledge_base/orchestrator.py', 'r') as f:
    content = f.read()

# Use index_elements instead of constraint with exclude
old_broken = """                # Upsert entities to avoid duplicate URI errors
                for entity in all_entities:
                    # Don't include updated_at - DB will set it via DEFAULT
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
                            'confidence': entity.confidence,
                            'source_text': entity.source_text
                        }
                    )
                    await session.execute(stmt)"""

new_working = """                # Use on_conflict_do_nothing to avoid the issue entirely
                for entity in all_entities:
                    # Skip if this entity already exists to avoid unique constraint errors
                    stmt = pg_insert(Entity).values(
                        id=entity.id,
                        name=entity.name,
                        entity_type=entity.entity_type,
                        description=entity.description,
                        properties=entity.properties,
                        confidence=entity.confidence,
                        uri=entity.uri,
                        source_text=entity.source_text
                    ).on_conflict_do_nothing()
                    await session.execute(stmt)"""

if old_broken in content:
    content = content.replace(old_broken, new_working)
    print("✅ Replaced with on_conflict_do_nothing")
else:
    print("❌ Could not find old pattern")
    print("Checking line count...")
    old_lines = old_broken.count('\n')
    print(f"Old pattern has {old_lines} lines")
    
    # Try with different line endings
    old_fixed = old_broken.replace('\n', '\r\n')
    if old_fixed in content:
        content = content.replace(old_fixed, new_working)
        print("✅ Fixed with CRLF")

with open('src/knowledge_base/orchestrator.py', 'w') as f:
    f.write(content)
