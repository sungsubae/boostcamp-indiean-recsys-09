from datetime import datetime
from fastapi import APIRouter, Depends
from typing import List, Union
from sqlalchemy.orm import Session
import requests

from core.database import get_db
from core.utils.steam_api import request_user_profile, request_user_history
from schemas.user import UserInDB
from schemas.history import HistoryInDB, UserHistoryList
from crud.user import get_user, add_user, update_user_update_time
from crud.history import get_user_history, add_user_history


router = APIRouter(prefix="/login")


@router.get("/{user_id}")
def user_login(userid: int, db: Session = Depends(get_db)) -> Union[UserInDB, bool]:
    # 유저가 DB에 없으면 유저 profile을 api로 가져와 DB에 저장
    _user = get_user(db, userid)
    if _user == False:
        _user_new = request_user_profile(userid)
        _user = add_user(db, _user_new)
    
    # 유저 history가 없거나 갱신한지 하루가 지났다면 api로 가져와 DB에 저장
    if _user.update_time == None or (_user.update_time - datetime.now()).days >= 1:
        _history_list_new = request_user_history(userid)
        _history_list = add_user_history(db, _history_list_new)
        _user = update_user_update_time(db, _user)
        # 인퍼런스 서버에 post로 요청을 날린 뒤 결과를 DB에 저장
        gameid_list = [h.gameid for h in _history_list]
        user_history_data = {"gameid_list": gameid_list}
        # response = requests.post("49.50.162.219:30001/recom", json=user_history_data)
        # rec_list = response.json()["products"][0]["gameid_list"]

    return True
