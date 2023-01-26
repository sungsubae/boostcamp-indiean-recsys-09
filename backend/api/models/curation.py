from sqlalchemy import Column, DateTime, ForeignKey, BigInteger, Integer, ARRAY, String
from sqlalchemy.orm import relationship, backref

from core.database import Base


class CurationTable(Base):
    __tablename__ = "curation"

    id = Column(BigInteger, primary_key=True, index=True)
    theme = Column(String)
    games = Column(ARRAY(Integer))
    time_created = Column(DateTime)
