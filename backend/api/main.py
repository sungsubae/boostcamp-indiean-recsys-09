import requests

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates

import pydantic
from typing import Optional

from core.setting import config
from routers import user


if config.env == "dev":
    import pandas as pd
    import numpy as np
    
    fake_user_history = pd.DataFrame(columns=["userid","appid","playtime_forever","rtime_last_played"])


app = FastAPI()
templates = Jinja2Templates(directory='./')
app.include_router(user.router)

@app.get("/")
def get_root():
    return {"Hello":"World"}


@app.get("/favicon.ico", include_in_schema=False)
def get_favicon():
    return FileResponse('asset/favicon.ico')


@app.get("/login")
def get_login_form(request: Request):
    return templates.TemplateResponse("login_form.html", context={"request": request})


async def request_user_history(userid: str):
    api_url = f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v1"
    data = {
        'key': config.steam.apikey,
        'steamid': userid,
        'include_played_free_games': True
    }
    response = requests.get(api_url,params=data).json()

    try:
        game_count = response["response"]["game_count"]
    except KeyError:
        raise HTTPException(status_code=401, detail="스팀 프로필의 게임 정보가 비공개입니다.")

    if game_count == 0:
        raise HTTPException(status_code=401, detail="스팀 게임 정보가 부족합니다.")
    
    user_history = [[game['appid'],
                    game['playtime_forever'],
                    game['rtime_last_played'],] 
                    for game in response['response']['games']]
    user_history = list(zip(*user_history))
    
    if sum(user_history[1]) == 0:
        raise HTTPException(status_code=401, detail="스팀 프로필의 총 플레이타임이 비공개입니다.")
    return pd.DataFrame({
        "userid": userid,
        "appid": user_history[0],
        "playtime_forever": user_history[1],
        "rtime_last_played": user_history[2],
    })

async def get_user_history(userid: str):
    global fake_user_history
    # userid's history exist in DB
    if (fake_user_history['userid']==userid).any():
        return fake_user_history[fake_user_history["userid"] == userid]
    
    # save and return new user's history
    new_user_history = await request_user_history(userid=userid)
    fake_user_history = pd.concat([fake_user_history, new_user_history], ignore_index=True)

    return new_user_history



@app.post("/login")
def login(userid: str = Form(...)):
    return {"userid": userid}

# post로 유저의 플레이정보를 인퍼런스 서버로 보내는데 이것은 client에서 get으로 요청해야 하는가 post로 요청해야 하는가... 그것이 문제로다.
@app.get("/recommendation/{userid}")
async def get_recommendation(userid: str):
    user_history =  await get_user_history(userid=userid)
    response = requests.post