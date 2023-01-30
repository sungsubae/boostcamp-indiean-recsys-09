from datetime import datetime

from sqlalchemy.orm import Session

from models import UserTable
from schemas.user import UserCreate


def get_user_list(db: Session):
    user_list = db.query(UserTable).order_by(UserTable.id.asc()).all()
    return user_list


def get_user(db: Session, user_id: int):
    u = db.query(UserTable).filter(UserTable.id == user_id).first()

    if u is None:
        return False
    else:
        return u


def add_user(db: Session, new_user: UserCreate):
    db_user = UserTable(
        id=new_user.id,
        persona_name=new_user.persona_name,
        update_time=None,
        recommend_time=None,
        time_created=new_user.time_created,
    )
    db.add(db_user)
    db.commit()
    return db_user


def update_user_update_time(db: Session, _user: UserTable):
    _user.update_time = datetime.utcnow()
    db.commit()
    return _user


def update_user_recommend_time(db: Session, _user: UserTable):
    _user.recommend_time = datetime.utcnow()
    db.commit()
    return _user