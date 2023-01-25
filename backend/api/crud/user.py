from sqlalchemy.orm import Session

from models import UserTable


def get_user_list(db:Session):
    user_list = db.query(UserTable)\
                .order_by(UserTable.id.asc())\
                .all()
    return user_list