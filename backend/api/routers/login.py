from datetime import datetime
from fastapi import APIRouter, Depends
from typing import List, Union
from sqlalchemy.orm import Session

from core.database import get_db
from core.steam_api import request_user_profile, request_user_history
from schemas.user import UserInDB
from schemas.history import HistoryInDB
from crud.user import get_user, add_user
from crud.history import get_user_history, add_user_history


router = APIRouter(prefix="/login")


@router.get("/{user_id}")
def user_login(userid: int, db: Session = Depends(get_db)) -> Union[UserInDB, bool]:
    _user = get_user(db, userid)
    # TODO 1: check if user is in db
    # TODO 2: if new user, get profile info api request and add to db
    if _user == False:
        _user = request_user_profile(userid)
        add_user(db, new_user)
    # TODO 3: if old user, check history update time.
    # TODO 4: for new user and old user 24 hours after update, get history
    # TODO 5: save new history and return
    if _user.update_time == None or (_user.update_time - datetime.now()).days >= 1:
        _history_list = request_user_history(userid)
        add_user_history(db, _history_list)
    # TODO 6: post inference and save to db
    return True
