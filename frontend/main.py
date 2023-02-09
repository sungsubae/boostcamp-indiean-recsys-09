from fastapi import FastAPI, Form, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
import requests
import uvicorn
import os
import typing


TEMP=[[1139890,'Dictators:No Peace Countryballs'],
    [1082710,'Bug Fables: The Everlasting Sapling'],
    [743390,'DISTRAINT 2'],
    [1717640,'Kill It With Fire'],
    [1293180,'SuchArt: Genius Artist Simulator'],
    [1018800,'DEEEER Simulator: Your Average Everyday Deer Game'],
    [1726400,'The Death | Thần Trùng'],
    [986800,'AVICII Invector'],
    [1164050,'When The Past Was Around'],
    [344770,'fault - milestone two side:above']]
middleware = [
 Middleware(SessionMiddleware, secret_key='super-secret')
]
app = FastAPI(middleware=middleware)
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR,"static")), name="static")
templates=Jinja2Templates(directory=os.path.join(BASE_DIR,"templates"))
server_url=""
@app.get("/login")
def loginpage(request: Request):
    islogin=False
    steam_id=None
    request.session['steamid']=None
    url="http://0.0.0.0:8000" #로그인 후 돌아올 경로
    domain="http://0.0.0.0" #도메인 영역
    
    if request.query_params:
        steam_id=request.query_params['openid.identity'][-17:]
        request.session['steamid']=request.query_params['openid.identity'][-17:]
        islogin=True
        param={"id":int(request.session['steamid'])}
        res=requests.post(url=f'{server_url}/login/checkdb',json=param)
        print("로그인 부분: ",res.status_code)
    return templates.TemplateResponse('login.html', context={"request":request,"islogin":islogin,"steam_id":steam_id,"url":url,"domain":domain})


@app.get("/index")
def index(request: Request):
    steam_id=request.session['steamid']
    if steam_id == None:
        rule_games=totalgame()
        return templates.TemplateResponse('history_false.html',context={"request":request,"isexist":False,"rule_games":rule_games})
    else:
        history=True #TODO API
        if history:
            res=requests.get(url=f'{server_url}/user/recommends/{steam_id}')
            recommends=res.json()['recommends']
            rec_games=[]
            for i in recommends:
                rec_games.append([i['game']['id'],i['game']['name']])
            
            return templates.TemplateResponse('history_true.html', context={"request":request,"isexist":True,"rec_games":rec_games})

        else:
        
            rule_games=totalgame()
            return templates.TemplateResponse('history_false.html', context={"request":request,"isexist":True,"rule_games":rule_games})


def totalgame():
    return TEMP