from sqlalchemy import Column, DateTime, ForeignKey, BigInteger, Integer, ARRAY
from sqlalchemy.orm import relationship, backref

from core.database import Base


class RecommendTable(Base):
    __tablename__ = "recommend"

    id = Column(BigInteger, primary_key=True, index=True)
    userid = Column(BigInteger, ForeignKey("users.id"))
    gameid = Column(Integer, ForeignKey("game.id"))
    time_created = Column(DateTime)

    user = relationship("UserTable", backref=backref("recommends"))
    game = relationship("GameTable", backref=backref("recommends"))