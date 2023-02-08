from fastapi import FastAPI, Form, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
import requests
import uvicorn
import os
import typing


TEMP=[[1680880,'Forspoken'],
                           [1809700,'Persona 3 Portable'],
                           [1036240,'Definitely Not Fried Chicken'],
                           [1172470,'Apex Legends'],
                           [1717640,'Mahokenshi'],
                           [271590,'Grand Theft Auto V'],
                           [431960,'Wallpaper Engine'],
                           [440,'Team Fortress 2'],
                           [1938090,'Call of Duty®: Modern Warfare® II | Warzone™ 2.0'],
                           [1203220,'NARAKA: BLADEPOINT'],
                           [730,'Counter-Strike: Global Offensive'],
                           [570,'Dota 2'],
                           [578080,'PUBG: BATTLEGROUNDS'],
                           [1172470,'Apex Legends'],
                           [1568590,'Goose Goose Duck'],
                           [271590,'Grand Theft Auto V'],
                           [431960,'Wallpaper Engine'],
                           [440,'Team Fortress 2'],
                           [1938090,'Call of Duty®: Modern Warfare® II | Warzone™ 2.0'],
                           [1203220,'NARAKA: BLADEPOINT']]
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