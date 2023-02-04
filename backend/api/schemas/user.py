from datetime import datetime

from pydantic import BaseModel, Field, validator
from typing import List, Optional

from schemas.history import HistoryGameDB
from schemas.recommend import RecommendInfoDB


class UserInDB(BaseModel):
    id: int
    persona_name: str
    update_time: Optional[datetime]
    recommend_time: Optional[datetime]
    time_created: datetime
    
    class Config:
        orm_mode = True


class UserRecommendsDB(BaseModel):
    recommend_time: Optional[datetime]
    recommends: List[RecommendInfoDB]

    class Config:
        orm_mode = True


class UserHistoriesDB(BaseModel):
    id: int
    persona_name: str
    histories: List[HistoryGameDB]

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    id: int


class User(BaseModel):
    id: int
    persona_name: str
    update_time: Optional[datetime]
    recommend_time: Optional[datetime]
    time_created: datetime
    
    @validator('id','persona_name')
    def not_none(cls, v):
        if v == None:
            raise ValueError(f"{cls}에 빈 값은 허용되지 않습니다.")
        return v


class UserCreate(BaseModel):
    id: int
    persona_name: str
    update_time: Optional[datetime]
    recommend_time: Optional[datetime]
    time_created: datetime
    
    @validator('id','persona_name')
    def not_none(cls, v):
        if v == None:
            raise ValueError(f"{cls}에 빈 값은 허용되지 않습니다.")
        return v


class UserList(BaseModel):
    user_list: List[User] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.user_list)