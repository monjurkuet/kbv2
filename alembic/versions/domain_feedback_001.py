"""Migration to add domain_detection_feedback table.

Revision ID: domain_feedback_001
Revises: experience_bank_001
Create Date: 2026-02-08
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "domain_feedback_001"
down_revision = "experience_bank_001"
branch_labels = None
depends_on = None


def upgrade():
    # Create domain_detection_feedback table
    op.create_table(
        "domain_detection_feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id", postgresql.UUID(as_uuid=True), nullable=False, index=True
        ),
        sa.Column("detected_domain", sa.String(), nullable=False, index=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("detection_method", sa.String(), nullable=True),
        sa.Column("crypto_indicators", postgresql.JSONB(), default=list),
        sa.Column("domain_scores", postgresql.JSONB(), default=dict),
        sa.Column("user_correction", sa.String(), nullable=True),
        sa.Column("feedback_source", sa.String(), nullable=True),
        sa.Column("extraction_quality", sa.Float(), nullable=True),
        sa.Column("entity_count", sa.Integer(), nullable=True),
        sa.Column("was_accurate", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )

    # Create indexes
    op.create_index(
        "ix_feedback_domain_accuracy",
        "domain_detection_feedback",
        ["detected_domain", "was_accurate"],
    )
    op.create_index(
        "ix_feedback_confidence", "domain_detection_feedback", ["confidence"]
    )
    op.create_index(
        "ix_feedback_created_at", "domain_detection_feedback", ["created_at"]
    )


def downgrade():
    op.drop_index("ix_feedback_created_at", table_name="domain_detection_feedback")
    op.drop_index("ix_feedback_confidence", table_name="domain_detection_feedback")
    op.drop_index("ix_feedback_domain_accuracy", table_name="domain_detection_feedback")
    op.drop_table("domain_detection_feedback")
