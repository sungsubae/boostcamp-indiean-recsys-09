from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
import typing
import uvicorn

def flash(request: Request, message: typing.Any) -> None:
   if "_messages" not in request.session:
       request.session["_messages"] = []
       request.session["_messages"].append({"message": message})
def get_flashed_messages(request: Request):
   print(request.session)
   return request.session.pop("_messages") if "_messages" in request.session else []

middleware = [
 Middleware(SessionMiddleware, secret_key='super-secret')
]
app = FastAPI(middleware=middleware)

app.mount("/static/", StaticFiles(directory='static', html=True), name="static")
templates = Jinja2Templates(directory="templates")
templates.env.globals['get_flashed_messages'] = get_flashed_messages


@app.get("/login/", response_class=HTMLResponse)
async def login_form(request: Request):
   return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login/", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
   if username == "test" and password == "test":
       flash(request, "Login Successful")
       return templates.TemplateResponse("login.html", {"request": request})
   flash(request, "Failed to login")
   return templates.TemplateResponse("login.html", {"request": request})


