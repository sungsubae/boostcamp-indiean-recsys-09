import pandas as pd
import pathlib
from fastapi import APIRouter, Depends, Query
from pydantic import Field
from typing import List, Union
from sqlalchemy.orm import Session

from core.database import get_db
from schemas.user import UserInDB
from schemas.game import GameInDB
from schemas.curation import CurationBase, CurationList
from crud import game


PATH = pathlib.Path(__file__).parent.parent.resolve()
router = APIRouter(prefix="/game")


@router.get("/info/{gameid}", response_model=GameInDB)
def game_info(gameid: int, db: Session=Depends(get_db)):
    _game = game.get_game(db, gameid)
    return _game


def get_csv():
    filtered_df = pd.read_csv(PATH/'assets'/'filtered.csv')
    return filtered_df


@router.get("/selection")
def game_selection(
    price: int=Query(default=0, ge=0, le=4000),
    genre: List[str]=Query(default=[]),
    category: List[str]=Query(default=[]),
    platform: List[str]=Query(default=[]),
    df: pd.DataFrame=Depends(get_csv)):

    if genre:
        df = df[df['Genre'].str.contains('&'.join(genre), na=False)]

    # category filtering
    if category:
        df = df[df['Categories'].str.contains('&'.join(category), na=False)]

    # price filtering
    if price:
        df = df[df['Initial Price'] <= price]

    # platform filtering
    if platform:
        df = df[df['Platforms'].str.contains('&'.join(platform), na=False)]


    N = min(10, len(df))
    if N == 0:
        return 0
    else:
        sample_df = df.sample(n=N)
        result = [
            CurationBase(gameid=gid, name=name)
            for gid,name in zip(sample_df['App ID'].values, sample_df['Name'].values)
        ]
        return CurationList(curations=result)