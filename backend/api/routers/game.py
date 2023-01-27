from fastapi import APIRouter, Depends
from typing import List, Union
from sqlalchemy.orm import Session

from core.database import get_db
from schemas.user import UserInDB
from crud import game


router = APIRouter(prefix="/game")


@router.get("/recommendation/{userid}")
def recommendation_list(db: Session = Depends(get_db)) -> List[UserInDB]:
    _user_list = user.get_user_list(db)
    return _user_list


@router.get("/curation")
def user_by_id(user_id: int, db: Session = Depends(get_db)) -> Union[UserInDB, bool]:
    _user = user.get_user(db, user_id)
    return _user
