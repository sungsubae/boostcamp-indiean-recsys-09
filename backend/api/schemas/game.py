from datetime import datetime

from pydantic import BaseModel, Field, validator
from typing import List


class GameInDB(BaseModel):
    id : int
    name : str
    decription : str
    developer : str
    publisher : str
    genres : str
    tags : str
    types : str
    categories : str
    owners : str
    positive_review : int
    negative_review : int
    price : float
    initial_price : float
    discount : float
    ccu : int
    languages : str
    platforms : str
    release_date : datetime
    required_age : int
    header_image : str

    class Config:
        orm_mode = True


class Game(BaseModel):
    id : int
    name : str
    decription : str
    developer : str
    publisher : str
    genres : str
    tags : str
    types : str
    categories : str
    owners : str
    positive_review : int
    negative_review : int
    price : float
    initial_price : float
    discount : float
    ccu : int
    languages : str
    platforms : str
    release_date : datetime
    required_age : int
    header_image : str

    @validator("id","name")
    def not_none(cls, v):
        if not v:
            raise ValueError(f"{cls}에 빈 값은 허용되지 않습니다.")


class GameList(BaseModel):
    game_list: List[Game] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.game_list)