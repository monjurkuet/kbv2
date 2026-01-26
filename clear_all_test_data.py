import sys
sys.path.insert(0, 'src')
from sqlalchemy import create_engine, text

engine = create_engine("postgresql://agentzero@localhost:5432/knowledge_base")

with engine.connect() as conn:
    print("Clearing all test data...")
    
    # Delete in correct order to avoid FK violations
    conn.execute(text("DELETE FROM chunk_entities"))
    conn.execute(text("DELETE FROM edges"))
    conn.execute(text("DELETE FROM entities"))
    conn.execute(text("DELETE FROM chunks"))
    conn.execute(text("DELETE FROM documents"))
    
    conn.commit()
    print("✅ All test data cleared")
