import requests
from datetime import datetime

from fastapi import HTTPException

from schemas.user import UserCreate
from schemas.history import HistoryCreate
from core.database import config


def request_user_profile(userid: int):
    api_url = "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2"
    data = {
        "key": config.steam.apikey,
        "steamids": userid,
    }
    response = requests.get(api_url, params=data).json()
    response = response["response"]["players"][0]
    print(response)
    return UserCreate(
        id=userid,
        persona_name=response["personaname"],
        time_created=datetime.fromtimestamp(int(response["timecreated"])),
    )


def request_user_history(userid: int):
    api_url = f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v1"
    data = {
        "key": config.steam.apikey,
        "steamid": userid,
        "include_played_free_games": True,
    }
    response = requests.get(api_url, params=data).json()

    try:
        game_count = response["response"]["game_count"]
    except KeyError:
        raise HTTPException(status_code=401, detail="스팀 프로필의 게임 정보가 비공개입니다.")

    if game_count == 0:
        raise HTTPException(status_code=401, detail="스팀 게임 정보가 부족합니다.")

    user_history = [
        [
            game["appid"],
            game["playtime_forever"],
            game["rtime_last_played"],
        ]
        for game in response["response"]["games"]
    ]
    user_history = list(zip(*user_history))

    if sum(user_history[1]) == 0:
        raise HTTPException(status_code=401, detail="스팀 프로필의 총 플레이타임이 비공개입니다.")

    return [
        HistoryCreate(
            userid=userid,
            gameid=g["appid"],
            playtime_total=g["playtime_forever"],
            rtime_last_played=g["rtime_last_played"],
        )
        for g in response["response"]["games"]
    ]
