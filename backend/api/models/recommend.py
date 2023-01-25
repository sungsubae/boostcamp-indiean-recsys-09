from sqlalchemy import Column, DateTime, ForeignKey, BigInteger, Integer, ARRAY
from sqlalchemy.orm import relationship

from core.database import Base


class RecommendTable(Base):
    __tablename__ = "recommend"

    id = Column(BigInteger, primary_key=True, index=True)
    userid = Column(BigInteger, ForeignKey("users.id"))
    games = Column(ARRAY(Integer))
    time_created = Column(DateTime)