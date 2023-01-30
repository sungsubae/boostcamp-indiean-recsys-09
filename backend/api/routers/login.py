from datetime import datetime
from fastapi import APIRouter, Depends
from typing import List, Union
from sqlalchemy.orm import Session

from core.database import get_db
from core.utils.steam_api import request_user_profile, request_user_history
from core.utils.inference import request_recommendation
from schemas.user import UserInDB
from crud.user import get_user, add_user, update_user_update_time, update_user_recommend_time
from crud.history import upsert_user_history
from crud.recommend import delete_and_add_recommends


router = APIRouter(prefix="/login")


@router.get("/{user_id}")
def user_login(userid: int, db: Session = Depends(get_db)) -> Union[UserInDB, bool]:
    # 유저가 DB에 없으면 유저 profile을 api로 가져와 DB에 저장
    _user = get_user(db, userid)
    if _user == False:
        _user_new = request_user_profile(userid)
        _user = add_user(db, _user_new)
    
    # 유저 history가 없거나 갱신한지 하루가 지났다면 api로 가져와 DB에 저장
    if _user.update_time == None or (_user.update_time - datetime.utcnow()).days >= 1:
        _history_list = request_user_history(userid)
        upsert_user_history(db, _history_list)
        _user = update_user_update_time(db, _user)
        # 인퍼런스 서버에 post로 요청을 날린 뒤 결과를 DB에 저장
    
    # 추천결과가 없거나 갱신한지 하루가 지났다면 api로 가져와 DB에 저장
    if _user.recommend_time == None or (_user.recommend_time - datetime.utcnow()).days >= 1:
        gameid_list = [h.gameid for h in _history_list]
        new_rec_list = request_recommendation(userid, gameid_list)
        delete_and_add_recommends(db, userid, new_rec_list)
        _user = update_user_recommend_time(db,_user)

    return True
