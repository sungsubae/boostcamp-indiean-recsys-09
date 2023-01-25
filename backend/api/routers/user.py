from fastapi import APIRouter, Depends
from typing import List
from sqlalchemy.orm import Session

from core.database import get_db
from schemas.user import UserInDB
from crud import user


router = APIRouter(
    prefix="/user"
)


@router.get("/list")
def user_list(db: Session=Depends(get_db)) -> List[UserInDB]:
    _user_list = user.get_user_list(db)
    return _user_list