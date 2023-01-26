from datetime import datetime

from sqlalchemy.orm import Session
from typing import List

from models import GameTable


def get_game(db:Session, gameid:int):
    _game = db.query(GameTable)\
        .filter(GameTable.id==gameid)\
        .first()
    if _game is None:
        return False
    else:
        return _game


def get_game_list(db:Session, gameids:List[int]):
    _game_list = db.query(GameTable)\
        .filter(GameTable.id.in_(gameids))\
        .all()
    return _game_list