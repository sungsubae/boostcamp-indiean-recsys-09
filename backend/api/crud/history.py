from datetime import datetime

from sqlalchemy.orm import Session
from typing import List

from models import HistoryTable, UserTable
from schemas import user, history


def get_user_history(db: Session, q_user: user.UserInDB):
    latest_time = q_user.update_time
    userid = q_user.id
    history_list = (
        db.query(HistoryTable)
        .filter(HistoryTable.create_time == latest_time, HistoryTable.userid == userid)
        .all()
    )
    return history_list


def add_user_history(db: Session, new_history: List[history.HistoryCreate]):
    create_time = datetime.now()
    new_db_history = [HistoryTable(**h, create_time=create_time) for h in new_history]
    db.bulk_save_objects(new_db_history)
