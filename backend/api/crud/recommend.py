from datetime import datetime

from sqlalchemy.orm import Session
from typing import List

from models import RecommendTable
import schemas.recommend as rec_schem


def delete_recommends(db: Session, userid: int):
    db.query(RecommendTable).filter(RecommendTable.userid==userid).delete()
    db.commit()


def delete_and_add_recommends(db: Session, userid:int, new_recommends: List[rec_schem.RecommendCreate]):
    delete_recommends(db, userid)
    new_db_recommends = [RecommendTable(**r.dict()) for r in new_recommends]
    db.bulk_save_objects(new_db_recommends)
    db.commit()
    return new_db_recommends

