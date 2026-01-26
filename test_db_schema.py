#!/usr/bin/env python3
"""Check if we need to create a proper test document"""

from knowledge_base.persistence.v1.schema import Entity
from sqlalchemy import create_engine, text

engine = create_engine("postgresql://agentzero@localhost:5432/knowledge_base")

# Check if test document exists
print("Testing a simple insert to verify upsert works...")
print("="*70)

import asyncio
from datetime import datetime
from uuid import uuid4
from sqlalchemy import select

async def test_simple_insert():
    async def get_session():
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        return async_session()
    
    session = await get_session()
    
    # Test simple entity insert
    entity = Entity(
        id=uuid4(),
        name="Test Entity",
        entity_type="Test",
        confidence=0.95,
        uri="entity:test_entity",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()  # Set explicitly
    )
    
    try:
        async with session.begin():
            session.add(entity)
        print("✅ Simple insert works")
        return True
    except Exception as e:
        print(f"❌ Insert failed: {e}")
        return False

result = asyncio.run(test_simple_insert())
print("="*70)
sys.exit(0 if result else 1)
