from fastapi import APIRouter, Depends
from typing import List, Union
from sqlalchemy.orm import Session

from core.database import get_db
from schemas.user import UserInDB
from schemas.game import GameInDB
from crud import game


router = APIRouter(prefix="/game")


@router.get("/info/{gameid}", response_model=GameInDB)
def game_info(gameid: int, db: Session=Depends(get_db)):
    _game = game.get_game(db, gameid)
    return _game