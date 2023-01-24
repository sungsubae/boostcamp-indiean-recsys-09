from core.database import Base, engine
from models import UserTable, GameTable, HistoryTable

def _add_tables():
    return Base.metadata.create_all(bind=engine)