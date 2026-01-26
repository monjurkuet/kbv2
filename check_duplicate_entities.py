#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from knowledge_base.persistence.v1.schema import Entity

# Check the database
engine = create_engine("postgresql://agentzero@localhost:5432/knowledge_base")

with Session(engine) as session:
    # Check for existing entities with iPhone 15
    existing = session.query(Entity).filter(Entity.name.ilike('%iphone%')).all()
    
    print(f"Found {len(existing)} iPhone entities:")
    for e in existing:
        print(f"  - {e.name} (URI: {e.uri})")
        
    if existing:
        print(f"\n⚠️  Database already has iPhone entities - this will cause duplicate errors")
        print(f"\nSolution: Either:")
        print(f"  1. Delete existing: session.query(Entity).filter(...).delete()")
        print(f"  2. Use upsert instead of insert")
        print(f"  3. Clear database before test")
    else:
        print(f"✓ No iPhone entities found - should not get duplicate error")
