from fastapi import FastAPI
from fastapi.param_functions import Depends
from pydantic import BaseModel
from typing import List
from pandas import DataFrame

import pandas as pd


#################### app 선언 ####################
app = FastAPI()


#################### class 및 utils 정의 ####################
class filteringOption(BaseModel):
    genre: List[str]
    category: List[str]
    price: int
    platform: List[str]

class curationResult(BaseModel):
    item_id: List[int]
    item_title: List[str]

def get_csv():
    filtered_df = pd.read_csv('filtered.csv')
    return filtered_df

    
#################### Curation 서버 구축 ####################
orders = [] #  DB 수정 필요

@app.post("/curation", description = "큐레이션 필터링 옵션을 요청합니다.")
async def make_curation(input: filteringOption, df: DataFrame = Depends(get_csv)):
    # genre filtering
    if input.genre:
        df = df[df['Genre'].str.contains('&'.join(input.genre), na=False)]

    # category filtering
    if input.category:
        df = df[df['Categories'].str.contains('&'.join(input.category), na=False)]

    # price filtering
    if input.price:
        df = df[df['Initial Price'] <= input.price]

    # platform filtering
    if input.platform:
        df = df[df['Platforms'].str.contains('&'.join(input.platform), na=False)]


    N = min(10, len(df))
    if N == 0:
        return 0
    else:
        return curationResult(
            item_id=list(df.sample(n=N)['App ID'].values), 
            item_title=list(df.sample(n=N)['Name'].values))


#################### main ####################
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000) # reload=True

    

# Step1: 로컬 csv 사용하여 큐레이션 결과 프론트로 전달
# Step2: DB 연결하여 DB에서 데이터 가져와서 큐레이션 진행한 뒤 프론트로 전달