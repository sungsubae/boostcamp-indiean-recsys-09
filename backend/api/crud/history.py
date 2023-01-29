from datetime import datetime

from sqlalchemy import insert
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import Insert
from typing import List

from models import HistoryTable, UserTable
from schemas import user, history

def get_user_history(db: Session, q_user: user.UserInDB):
    latest_time = q_user.update_time
    userid = q_user.id
    history_list = (
        db.query(HistoryTable)
        .filter(HistoryTable.userid == userid)
        .all()
    )
    return history_list


def add_user_history(db: Session, new_history: List[history.HistoryCreate]) -> List[HistoryTable]:
    new_db_history = [HistoryTable(**h) for h in new_history]
    db.bulk_save_objects(new_db_history)
    db.commit()
    Insert.on_conflict_do_update()
    return new_db_history

# TODO: delete and add user history