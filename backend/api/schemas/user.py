from datetime import datetime

from pydantic import BaseModel, Field, validator
from typing import List


class UserInDB(BaseModel):
    id: int
    persona_name: str
    update_time: datetime
    time_created: datetime
    
    class Config:
        orm_mode = True


class User(BaseModel):
    id: int
    persona_name: str
    update_time: datetime
    time_created: datetime
    
    @validator('id','persona_name')
    def not_none(cls, v):
        if not v:
            raise ValueError(f"{cls}에 빈 값은 허용되지 않습니다.")


class UserList(BaseModel):
    user_list: List[User] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.user_list)