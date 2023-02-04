"""empty message

Revision ID: 4c62b9df2c60
Revises: 45be65cd0f6c
Create Date: 2023-01-30 04:09:16.056234

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '4c62b9df2c60'
down_revision = '45be65cd0f6c'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('recommend', 'time_created')
    op.add_column('users', sa.Column('recommend_time', sa.DateTime(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('users', 'recommend_time')
    op.add_column('recommend', sa.Column('time_created', postgresql.TIMESTAMP(), autoincrement=False, nullable=True))
    # ### end Alembic commands ###
