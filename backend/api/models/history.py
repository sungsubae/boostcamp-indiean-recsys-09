from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, TIMESTAMP, BigInteger, Float, SmallInteger
from sqlalchemy.orm import relationship, backref

from core.database import Base


class HistoryTable(Base):
    __tablename__ = "history"

    id = Column(BigInteger, primary_key=True, index=True)
    userid = Column(BigInteger, ForeignKey("users.id"))
    appid = Column(Integer, ForeignKey("game.id"))
    playtime_total = Column(Float)
    rtime_last_played = Column(DateTime)

    user = relationship("UserTable", backref=backref("history"))
    game = relationship("GameTable", backref=backref("history"))