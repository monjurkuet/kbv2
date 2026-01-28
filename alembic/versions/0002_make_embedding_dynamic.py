"""make_embedding_dynamic

Revision ID: 0002_make_embedding_dynamic
Revises: b20e59aa81e1
Create Date: 2026-01-29 00:00:00.000000

This migration changes embedding dimensions from fixed 768 to 1024
which is perfect for bge-m3 (exact match) and supports smaller models.

Note: pgvector IVFFlat index has a limit of 2000 dimensions.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0002_make_embedding_dynamic'
down_revision: Union[str, Sequence[str], None] = 'b20e59aa81e1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to support larger embedding dimensions."""
    print("Upgrading embedding dimensions from 768 to 1024...")
    
    # Drop existing indexes first
    op.drop_index('idx_entity_embedding_ivfflat', table_name='entities')
    op.drop_index('idx_chunk_embedding_ivfflat', table_name='chunks')
    
    # Drop the old columns with 768 dimensions
    op.drop_column('entities', 'embedding')
    op.drop_column('chunks', 'embedding')
    
    # Add new columns with 1024 dimensions (perfect for bge-m3)
    op.execute("""
        ALTER TABLE entities 
        ADD COLUMN embedding vector(1024)
    """)
    
    op.execute("""
        ALTER TABLE chunks 
        ADD COLUMN embedding vector(1024)
    """)
    
    # Recreate indexes on the new columns
    op.execute("""
        CREATE INDEX idx_entity_embedding_ivfflat 
        ON entities USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100)
    """)
    
    op.execute("""
        CREATE INDEX idx_chunk_embedding_ivfflat 
        ON chunks USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100)
    """)
    
    print("Successfully upgraded embedding dimensions to 1024")
    print("Perfect for bge-m3 model!")


def downgrade() -> None:
    """Downgrade schema back to 768 dimensions."""
    raise NotImplementedError(
        "Cannot downgrade embedding dimensions safely. "
        "All embedding data would be lost. "
        "Manual intervention required if you must downgrade."
    )
