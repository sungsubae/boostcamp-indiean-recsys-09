from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import requests
import os
app=FastAPI()
global steam_id

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR,"static")), name="static")
templates=Jinja2Templates(directory=os.path.join(BASE_DIR,"templates"))

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse('index.html', context={"request":request})

@app.get("/login")
def loginpage(request: Request):
    return templates.TemplateResponse('login.html', context={"request":request})

@app.get("/login2")
def login2(request: Request):
    steam_id=request.query_params['openid.identity'][-17:]
    return templates.TemplateResponse('login2.html', context={"request":request,"steam_id":steam_id})

@app.get("/mypage")
async def mypage(request: Request):
    return templates.TemplateResponse('mypage.html', context={"request":request})


@app.get("/userpage")
def get_login_form(request: Request):
    return templates.TemplateResponse('userpage.html', context={"request":request})

@app.get("/gamepage")
def get_login_form(request: Request):
    return templates.TemplateResponse('gamepage.html', context={"request":request})


################################################################################



if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
    
    
    
