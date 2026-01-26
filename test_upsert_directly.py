#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
exec(open('load_env.py').read())

import asyncio
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert as pg_insert
from knowledge_base.persistence.v1.schema import Entity
from uuid import uuid4

async def test_upsert():
    engine = create_engine("postgresql://agentzero@localhost:5432/knowledge_base")
    
    async with engine.connect() as conn:
        # Try to insert duplicate
        stmt = pg_insert(Entity).values(
            id=uuid4(),
            name='CEO',
            entity_type='Job Title',
            uri='entity:ceo'  # This already exists
        ).on_conflict_do_update(
            constraint='entities_uri_key',
            set_={'name': 'CEO - Updated'}
        )
        
        try:
            await conn.execute(stmt)
            await conn.commit()
            print("✅ Upsert worked - no error")
            return True
        except Exception as e:
            print(f"❌ Upsert failed: {e}")
            return False

result = asyncio.run(test_upsert())
print(f"\nUpsert works: {'✅ YES' if result else '❌ NO'}")
