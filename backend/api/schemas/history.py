from datetime import datetime

from pydantic import BaseModel, Field, validator
from typing import List


class HistoryInDB(BaseModel):
    id: int
    userid: int
    gameid: int
    playtime_total: float
    rtime_last_played: datetime
    create_time: datetime

    class Config:
        orm_mode = True


class History(BaseModel):
    userid: int
    gameid: int
    playtime_total: float
    rtime_last_played: datetime
    create_time: datetime

    @validator("userid", "gameid", "playtime_total")
    def not_none(cls, v):
        if not v:
            raise ValueError(f"{cls}에 빈 값은 허용되지 않습니다.")
        return v


class HistoryCreate(BaseModel):
    userid: int
    gameid: int
    playtime_total: float
    rtime_last_played: datetime

    @validator("userid", "gameid", "playtime_total")
    def not_none(cls, v):
        if not v:
            raise ValueError(f"{cls}에 빈 값은 허용되지 않습니다.")
        return v


class HistoryList(BaseModel):
    history_list: List[History] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.history_list)
