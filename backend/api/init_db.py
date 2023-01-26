import numpy as np
import pandas as pd
import pathlib

from core.database import Base, engine, get_db
from models import UserTable, GameTable, HistoryTable


def to_list(x):
    if type(x) is str:
        x = x.split(', ')
        return x
    return []


def _add_tables():
    return Base.metadata.create_all(bind=engine)


def _add_games():
    db = next(get_db())
    PATH = pathlib.Path(__file__).parent.resolve()
    games = pd.read_csv(PATH/"assets"/"processed.csv",sep=';',parse_dates=["Release Date"])
    array_col = ['genres','tags','categories','languages','platforms']
    for col in array_col:
        games[col] = games[col].apply(to_list)
    games = games.replace({np.NaN:None})

    rows = [GameTable(**r) for r in games.to_dict('records')]
    db = next(get_db())
    try:
        db.bulk_save_objects(rows)
    except:
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    _add_games()