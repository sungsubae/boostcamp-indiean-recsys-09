from datetime import datetime

from pydantic import BaseModel, Field, validator
from typing import List, Optional

from schemas.game import GameBaseDB


class CurationBase(BaseModel):
    gameid: int
    name: str


class CurationList(BaseModel):
    curations: List[CurationBase]

    def __len__(self):
        return len(self.curations)
