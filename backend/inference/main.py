from fastapi import FastAPI, Request, Form, Cookie, Security, HTTPException
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

from datetime import datetime
from pandas import DataFrame

<<<<<<< Updated upstream
from ml.Inference import inference
from ml.model import NeuMF, get_model

from ml.ease import dataload, get_user, inference, EASE
=======
from ml.ease import get_user, dataload, train_predict
>>>>>>> Stashed changes

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

# 'appid','playtime_forever', 'uesrid' -> userid, playtime_forever의 type 파악
# json으로 받아와서 class 정의
class RecSteamProduct(BaseModel):
    userid : int 
    playtime_forever : Optional[List] = None
    gameid_list :Optional[List] = None

class inferenceSteamProduct(Product):
    name: str = "inference_steam_product"
    gameid_list: Optional[List] = None
    
    
############# Inferenece 서버 구축###########
# orders = [] #  DB 수정 필요

<<<<<<< Updated upstream
@app.post("/recomNeu", description = "로그인 정보 요청합니다.")
async def make_order(input: RecSteamProduct,
                                    model: NeuMF=Depends(get_model)):  # model, config 정의 필요, load_model 필요
    products = []
    # TODO 1: Recommend List
    gameid_list = inference(model)
    # input 인자 이와 같이 명시해줘야 함
    product = inferenceSteamProduct(gameid_list = gameid_list)
    products.append(product)
    
    new_order = Order(products=products)
    
    return new_order

# flow : dataload, get_user, train_predict
# dataload : key.json file path
# get_user : input.userid, input.playtime_forever, input.gameid_list
# inference : 

@app.post("/recom", description = "로그인 정보 요청합니다.")
async def make_order(input: RecSteamProduct,
                                    ):  # model, config 정의 필요, load_model 필요
    products = []
    # TODO 1: Recommend List
    train, games = dataload()
    print(input)
    print(type(input.userid))
    print(type(input.playtime_forever))
    print(type(input.gameid_list))
    test = get_user(input.userid, input.playtime_forever, input.gameid_list)
    gameid_list = inference(train, test, games, EASE())
    # input 인자 이와 같이 명시
    product = inferenceSteamProduct(gameid_list = gameid_list)
=======
@app.post("/recom", description = "로그인 정보 요청합니다.")
async def make_order(input: RecSteamProduct):  # model, config 정의 필요, load_model 필요
    train, game = dataload()
    api =  "F644C6E2851DBDC938FF6FC70388451F"
    test = get_user(userid, api)
    products = []
    # TODO 1: Recommend List
    gameid_list = train_predict(train, test, game)
    # titles, images = get_model_rec(model = model, input_ids = input.games, top_k = input.top_k) #  model inference
    product = inferenceSteamProduct(gameid_list)
>>>>>>> Stashed changes
    products.append(product)
    
    new_order = Order(products=products)
    
    return new_order
    
# train, game = dataload()
# test = get_user(userid, api) 
# output = inference(train, test, game)