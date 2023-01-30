from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, TIMESTAMP, BigInteger, Float, SmallInteger
from sqlalchemy.orm import relationship

from core.database import Base


class UserTable(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, index=True)
    persona_name = Column(String)
    update_time = Column(DateTime)
    recommend_time = Column(DateTime)
    time_created = Column(DateTime)