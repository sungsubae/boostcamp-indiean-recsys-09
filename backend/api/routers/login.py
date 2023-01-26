from fastapi import APIRouter, Depends
from typing import List, Union
from sqlalchemy.orm import Session

from core.database import get_db
from schemas.user import UserInDB
from schemas.history import HistoryInDB
from crud.user import get_user, add_user


router = APIRouter(
    prefix="/login"
)


@router.get("/{user_id}")
def user_login(user_id: int, db: Session=Depends(get_db)) -> Union[UserInDB, bool]:
    _user = get_user(db, user_id)
    pass