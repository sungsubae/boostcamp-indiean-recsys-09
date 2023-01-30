from fastapi import FastAPI, Form, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware

import uvicorn
import os
import typing

middleware = [
 Middleware(SessionMiddleware, secret_key='super-secret')
]
app = FastAPI(middleware=middleware)

fake_db=dict()
fake_db['user']=[]
#appid, gamename, gameimageurl
fake_db['ruleBasedItems']=[[730,'Counter-Strike: Global Offensive'],
                           [570,'Dota 2'],
                           [578080,'PUBG: BATTLEGROUNDS'],
                           [1172470,'Apex Legends'],
                           [1568590,'Goose Goose Duck'],
                           [271590,'Grand Theft Auto V'],
                           [431960,'Wallpaper Engine'],
                           [440,'Team Fortress 2'],
                           [1938090,'Call of Duty®: Modern Warfare® II | Warzone™ 2.0'],
                           [1203220,'NARAKA: BLADEPOINT']]

fake_db['newItems']=[[1680880,'Forspoken'],
                           [1809700,'Persona 3 Portable'],
                           [1036240,'Definitely Not Fried Chicken'],
                           [1172470,'Apex Legends'],
                           [1717640,'Mahokenshi'],
                           [271590,'Grand Theft Auto V'],
                           [431960,'Wallpaper Engine'],
                           [440,'Team Fortress 2'],
                           [1938090,'Call of Duty®: Modern Warfare® II | Warzone™ 2.0'],
                           [1203220,'NARAKA: BLADEPOINT']]
fake_db['recResults']=[[431960,'Wallpaper Engine'],
                           [440,'Team Fortress 2'],
                           [1938090,'Call of Duty®: Modern Warfare® II | Warzone™ 2.0'],
                           [1203220,'NARAKA: BLADEPOINT'],
                           [1568590,'Goose Goose Duck'],
                           [271590,'Grand Theft Auto V'],
                           [431960,'Wallpaper Engine'],
                           [440,'Team Fortress 2'],
                           [1938090,'Call of Duty®: Modern Warfare® II | Warzone™ 2.0'],
                           [1203220,'NARAKA: BLADEPOINT']]

def flash(request: Request, message: typing.Any) -> None:
   if "_messages" not in request.session:
       request.session["_messages"] = []
       request.session["_messages"].append({"message": message})
def get_flashed_messages(request: Request):
   print("session",request.session)
   return request.session.pop("_messages") if "_messages" in request.session else []

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR,"static")), name="static")
templates=Jinja2Templates(directory=os.path.join(BASE_DIR,"templates"))
templates.env.globals['get_flashed_messages'] = get_flashed_messages

@app.get("/index")
def index(request: Request):
    #TODO DB
    print(request.session)
    rule_based=fake_db['ruleBasedItems']
    new_games=fake_db['newItems']

    return templates.TemplateResponse('index.html', context={"request":request,"rule_based":rule_based,"new_games":new_games})


@app.get("/login")
async def loginpage(request: Request):
    #프로필 공개여부 계속 확인하다가 공개가 되면 유저정보 받아오기 + db에 저장
    islogin=False
    steam_id=None
    if request.query_params:
        steam_id=request.query_params['openid.identity'][-17:]
        request.session['steamid']=steam_id
        islogin=True
        #TODO if isopen이면 유저 프로필 받아와서 db에 값 넘겨주기
        fake_db['user'].append(steam_id)
        print(fake_db['user'])
    return templates.TemplateResponse('login.html', context={"request":request,"islogin":islogin,"steam_id":steam_id})

@app.post("/login")
async def loginpage(request:Request):
    print("STEAM ID: ",request.session['steamid'])
    ispulic=False
    if not ispulic:
        flash(request,"public please")
        return templates.TemplateResponse('login.html', context={"request":request,"ispulic":ispulic})
    return templates.TemplateResponse('login.html', context={"request":request,"ispulic":ispulic})


@app.post("/mypage/{user_id}")
def mypage(request: Request):
    
    return templates.TemplateResponse('mypage.html', context={"request":request})


@app.post("/userpage/{user_id}")
def userpage(request: Request,user_id):
    #TODO DB
    rec_results=fake_db['recResults']
    return templates.TemplateResponse('userpage.html', context={"request":request,"user_id":user_id,"rec_results":rec_results})

@app.get("/gamepage/{game_id}")
def gamepage(request: Request):
    return templates.TemplateResponse('gamepage.html', context={"request":request})

@app.post("/gamepage/{game_id}")
def gamepage(request: Request,user_id,game_id):
    
    print(f"{user_id},{game_id}")
    
    return templates.TemplateResponse('gamepage.html', context={"request":request})

    
