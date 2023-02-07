from datetime import datetime

from pydantic import BaseModel, Field, validator
from typing import List, Optional


class GameBaseDB(BaseModel):
    id: int
    name: str

    class Config:
        orm_mode = True


class GameInDB(BaseModel):
    id : int
    name : str
    description : Optional[str]
    developer : Optional[str]
    publisher : Optional[str]
    genres : Optional[List[str]]
    tags : Optional[List[str]]
    categories : Optional[List[str]]
    positive_review : Optional[int]
    negative_review : Optional[int]
    price : Optional[float]
    initial_price : Optional[float]
    discount : Optional[float]
    languages : Optional[List[str]]
    platforms : Optional[List[str]]
    release_date : Optional[datetime]
    required_age : Optional[str]
    header_image : Optional[str]

    class Config:
        orm_mode = True


class Game(BaseModel):
    id : int
    name : str
    description : Optional[str]
    developer : Optional[str]
    publisher : Optional[str]
    genres : Optional[List[str]]
    tags : Optional[List[str]]
    categories : Optional[List[str]]
    positive_review : Optional[int]
    negative_review : Optional[int]
    price : Optional[float]
    initial_price : Optional[float]
    discount : Optional[float]
    languages : Optional[List[str]]
    platforms : Optional[List[str]]
    release_date : Optional[datetime]
    required_age : Optional[str]
    header_image : Optional[str]

    @validator("id","name")
    def not_none(cls, v):
        if v == None:
            raise ValueError(f"{cls}에 빈 값은 허용되지 않습니다.")
        return v


class GameList(BaseModel):
    game_list: List[Game] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.game_list)