"""stratified calibration sampling

Revision ID: 34ea730b95fd
Revises: 8e1c9915624e
Create Date: 2026-11-12 12:16:40.263268

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '34ea730b95fd'
down_revision: Union[str, Sequence[str], None] = '8e1c9915624e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Existing rows came from a simple random sample: no stratum, weight 1.
    # The server default backfills them, then comes off so the column matches
    # the model, which carries a Python-side default instead.
    op.add_column(
        "review_items",
        sa.Column("stratum", sa.String(length=16), nullable=False, server_default=""),
    )
    op.add_column(
        "review_items",
        sa.Column("weight", sa.Float(), nullable=False, server_default="1.0"),
    )
    op.alter_column("review_items", "stratum", server_default=None)
    op.alter_column("review_items", "weight", server_default=None)


def downgrade() -> None:
    op.drop_column("review_items", "weight")
    op.drop_column("review_items", "stratum")