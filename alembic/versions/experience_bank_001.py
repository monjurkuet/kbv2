"""Create extraction_experiences table for Experience Bank.

Revision ID: experience_bank_001
Revises:
Create Date: 2026-02-07

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "experience_bank_001"
down_revision: Union[str, None] = "0003_upgrade_to_1024_dimensions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create extraction_experiences table
    op.create_table(
        "extraction_experiences",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("text_snippet", sa.Text(), nullable=False),
        sa.Column("text_embedding_id", sa.String(), nullable=True),
        sa.Column("entities", postgresql.JSONB(astext_type=sa.Text()), default=list),
        sa.Column(
            "relationships", postgresql.JSONB(astext_type=sa.Text()), default=list
        ),
        sa.Column(
            "extraction_patterns", postgresql.JSONB(astext_type=sa.Text()), default=dict
        ),
        sa.Column("domain", sa.String(), nullable=False, index=True),
        sa.Column(
            "entity_types", postgresql.JSONB(astext_type=sa.Text()), default=list
        ),
        sa.Column("quality_score", sa.Float(), nullable=False, index=True),
        sa.Column("extraction_method", sa.String(), nullable=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("retrieval_count", sa.Integer(), default=0),
        sa.Column("last_retrieved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), default=sa.func.now()),
    )

    # Create indexes
    op.create_index(
        "ix_experiences_domain_quality",
        "extraction_experiences",
        ["domain", "quality_score"],
    )

    op.create_index(
        "ix_experiences_entity_types",
        "extraction_experiences",
        ["entity_types"],
        postgresql_using="gin",
    )

    op.create_index(
        "ix_experiences_text_embedding", "extraction_experiences", ["text_embedding_id"]
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index("ix_experiences_text_embedding", table_name="extraction_experiences")
    op.drop_index("ix_experiences_entity_types", table_name="extraction_experiences")
    op.drop_index("ix_experiences_domain_quality", table_name="extraction_experiences")

    # Drop table
    op.drop_table("extraction_experiences")
