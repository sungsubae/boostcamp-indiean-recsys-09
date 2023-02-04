from fastapi import APIRouter, Depends
from typing import List, Union
from sqlalchemy.orm import Session

from core.database import get_db
from schemas.user import UserInDB, UserHistoriesDB, UserRecommendsDB
from crud import user


router = APIRouter(
    prefix="/user"
)


@router.get("/list")
def user_list(db: Session=Depends(get_db)) -> List[UserInDB]:
    _user_list = user.get_user_list(db)
    return _user_list


@router.get("/{userid}")
def user_by_id( user_id: int, db: Session=Depends(get_db)) -> Union[UserInDB, bool]:
    _user = user.get_user(db, user_id)
    return _user


@router.get("/histories/{userid}", response_model=UserHistoriesDB)
def user_history(userid:int, db: Session=Depends(get_db)):
    _user = user.get_user(db, userid)
    return _user


@router.get("/recommends/{userid}", response_model=UserRecommendsDB)
def user_history(userid:int, db: Session=Depends(get_db)):
    _user = user.get_user(db, userid)
    return _user