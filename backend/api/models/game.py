from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, TIMESTAMP, BigInteger, Float, SmallInteger
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY
from core.database import Base


class GameTable(Base):
    __tablename__ = "game"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    decription = Column(String)
    developer = Column(String)
    publisher = Column(String)
    genres = Column(String)
    tags = Column(String)
    types = Column(String)
    categories = Column(String)
    owners = Column(String)
    positive_review = Column(Integer)
    negative_review = Column(Integer)
    price = Column(Float)
    initial_price = Column(Float)
    discount = Column(Float)
    ccu = Column(Integer)
    languages = Column(String)
    platforms = Column(String)
    release_date = Column(DateTime)
    required_age = Column(SmallInteger)
    header_image = Column(String)
