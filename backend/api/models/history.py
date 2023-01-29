from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, TIMESTAMP, BigInteger, Float, SmallInteger
from sqlalchemy.orm import relationship, backref

from core.database import Base


class HistoryTable(Base):
    __tablename__ = "history"

    userid = Column(BigInteger, ForeignKey("users.id"), primary_key=True)
    gameid = Column(Integer, ForeignKey("game.id"), primary_key=True)
    playtime_total = Column(Float)
    rtime_last_played = Column(DateTime)

    user = relationship("UserTable", backref=backref("histories"))
    game = relationship("GameTable", backref=backref("histories"))