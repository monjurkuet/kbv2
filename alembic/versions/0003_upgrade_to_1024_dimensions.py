"""upgrade_to_1024_dimensions

Revision ID: 0003_upgrade_to_1024_dimensions
Revises: 0002_make_embedding_dynamic
Create Date: 2026-01-29 01:30:00.000000

Upgrade embedding dimensions from 768 to 1024 to support bge-m3 model.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0003_upgrade_to_1024_dimensions'
down_revision: Union[str, Sequence[str], None] = '0002_make_embedding_dynamic'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade embedding columns from 768 to 1024 dimensions."""
    print("Upgrading embedding dimensions from 768 to 1024 for bge-m3 support...")
    
    # Drop existing indexes
    op.drop_index('idx_entity_embedding_ivfflat', table_name='entities')
    op.drop_index('idx_chunk_embedding_ivfflat', table_name='chunks')
    
    # Drop the old columns
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
    
    # Recreate indexes on new columns
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
    
    print("Successfully upgraded to 1024 dimensions!")


def downgrade() -> None:
    """Downgrade to 768 dimensions."""
    raise NotImplementedError("Cannot downgrade - would lose data")
