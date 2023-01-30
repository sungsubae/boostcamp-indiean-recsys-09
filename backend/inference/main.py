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

from ml.Inference import inference
from ml.model import NeuMF, get_model

app = FastAPI()

'''

Login Part -> Backend Part로 전향


'''

############################class 정의 ###################################

class Product(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str

class Order(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    products: List[Product] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at:datetime = Field(default_factory=datetime.now)

# Backdend에서 User game list 갖고와서 여기에 전달
# json으로 받아와서 class 정의
class RecSteamProduct(BaseModel):
    # games: List[int]
    # top_k: int # 필요할까?
    gameid_list :Optional[List] = None

class inferenceSteamProduct(Product):
    name: str = "inference_steam_product"
    gameid_list: Optional[List] = None
    
    
############# Inferenece 서버 구축###########
# orders = [] #  DB 수정 필요

@app.post("/recom", description = "로그인 정보 요청합니다.")
async def make_order(input: RecSteamProduct,
                                    model: NeuMF=Depends(get_model)):  # model, config 정의 필요, load_model 필요
    products = []
    # TODO 1: Recommend List
    gameid_list = inference(model)
    # titles, images = get_model_rec(model = model, input_ids = input.games, top_k = input.top_k) #  model inference
    product = inferenceSteamProduct(gameid_list)
    products.append(product)
    
    new_order = Order(products=products)
    # orders.append(new_order)    # Need to Update from orders -> DB
    
    return new_order
    
