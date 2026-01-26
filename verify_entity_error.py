#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
from sqlalchemy import create_engine, text

engine = create_engine("postgresql://agentzero@localhost:5432/knowledge_base")

with engine.connect() as conn:
    # Check current entity count
    result = conn.execute(text("SELECT COUNT(*) FROM entities"))
    count = result.scalar()
    print(f"Current entity count: {count}")
    
    # Check for CEO entity (commonly causes duplicates)
    result = conn.execute(text("SELECT id, uri FROM entities WHERE uri = 'entity:ceo'"))
    ceo = result.fetchone()
    if ceo:
        print(f"CEO entity exists: {ceo.id} - {ceo.uri}")
    else:
        print("CEO entity not found - good for testing")

