from fastapi import FastAPI, Request, Form, Cookie, Security, HTTPException
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

### DB 관련 모듈 ###
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
# import backend.app.DB.crud as crud
# import backend.app.DB.schemas as schemas
# from backend.app.DB.database import SessionLocal, engine
# import backend.app.DB.models as models
#################

from fastapi.templating import Jinja2Templates

from datetime import datetime
from pandas import DataFrame

from ml.inference import NeuMF, get_model, get_user, get_model_rec_prototype, inference_

############################################################
############### model & inference import 필요 #####################


############################################################

app = FastAPI()
templates = Jinja2Templates(directory = 'frontend/templates')

##############Login#########################################

@app.get("/login/")
def get_login_form(request: Request):
    return templates.TemplateResponse('login_form.html', context = {"request" : request})

@app.post("/login/")
def login(username: str = Form(...), password : str = Form(...)):
    return {"username": username}

##############class 정의 및 ###################################

class Product(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str

class Order(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    products: List[Product] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at:datetime = Field(default_factory=datetime.now)

# front에서 User game list 갖고와서 여기에 전달
# json으로 받아와서 class 정의
class RecSteamProduct(BaseModel):
    # games: List[int]
    # top_k: int # 필요할까?
    gameid_list :Optional[List] = None

class inferenceSteamProduct(Product):
    name: str = "inference_steam_product"
    appids: Optional[List] = None
    
    
############# Inferenece 서버 구축###########
orders = [] #  DB 수정 필요

@app.post("/recom", description = "주문을 요청합니다.")
# TODO 
async def make_order(input: RecSteamProduct,
                                    model: NeuMF=Depends(get_model)):  # model, config 정의 필요, load_model 필요
    products = []
    # Only prototype
    titles, images = get_model_rec_prototype(get_user, model, inference_)
    # titles, images = get_model_rec(model = model, input_ids = input.games, top_k = input.top_k) #  model inference
    product = inferenceSteamProduct(title = titles, images = images)
    products.append(product)
    
    new_order = Order(products=products)
    orders.append(new_order)                    # Need to Update from orders -> DB
    
    return new_order
    
    