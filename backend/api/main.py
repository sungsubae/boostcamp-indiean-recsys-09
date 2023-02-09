import pathlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from core.setting import config
from routers import user, login, game

PATH = pathlib.Path(__file__).parent.resolve()
app = FastAPI()
app.include_router(user.router)
app.include_router(login.router)
app.include_router(game.router)

# front http://ip:port 추가
origins = []
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def get_root():
    return {"Hello": "World"}


@app.get("/favicon.ico", include_in_schema=False)
def get_favicon():
    return FileResponse(PATH / "assets" / "favicon.ico")