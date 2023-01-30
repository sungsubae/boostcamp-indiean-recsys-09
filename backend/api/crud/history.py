from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
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


def add_user_history(db: Session, new_history: List[history.HistoryCreate]):
    new_db_history = [HistoryTable(**h.dict()) for h in new_history]
    db.bulk_save_objects(new_db_history)
    db.commit()
    return new_db_history


# TODO: update or insert user history
def upsert_user_history(db: Session, new_history: List[history.HistoryCreate]):
    values = [_h.dict() for _h in new_history]
    stmt = insert(HistoryTable).values(values)
    primary_keys = ['userid','gameid']
    update_cols = {
        col.name: col for col in stmt.excluded
        if col.name not in primary_keys
    }
    stmt = stmt.on_conflict_do_update(
        index_elements=[HistoryTable.userid, HistoryTable.gameid],
        set_=update_cols
    )
    db.execute(stmt)
    db.commit()