from datetime import datetime

from sqlalchemy.orm import Session

from models import UserTable
from schemas.user import UserCreate


def get_user_list(db:Session):
    user_list = db.query(UserTable)\
                .order_by(UserTable.id.asc())\
                .all()
    return user_list

def get_user(db:Session, user_id: int):
    u = db.query(UserTable)\
        .filter(UserTable.id == user_id)\
        .all()
    
    if len(u) == 0:
        return False
    else:
        return u[0]

def add_user(db:Session, new_user:UserCreate):
    db_user = UserTable(id=new_user.id,
                        persona_name=new_user.persona_name,
                        update_time=datetime.now(),
                        time_created=new_user.time_created)
    db.add(db_user)