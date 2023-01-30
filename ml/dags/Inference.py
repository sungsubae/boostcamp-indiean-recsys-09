# 새 유저에 대한 Inference를 위한 파일입니다
# inference 대상에 대한 것을 steam api로 가져와 예측합니다
# input >> user id // output >> pred app id
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import joblib
import os
import pandas as pd
import time
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import joblib
import requests

import warnings
warnings.filterwarnings("ignore")


from Train import NeuMF

def dataload():
    credential_path = 'key.json'
    credentials = service_account.Credentials.from_service_account_file(credential_path)
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    q="select i.i from (select distinct item_id as i from `data.interaction`) as i inner join (select distinct app_id as i from `data.game`) as g on i.i=g.i"
    origin = client.query(q).to_dataframe()
    n_items = origin.shape[0]
    
    q1 = "select App_ID,Tags, Name from `data.game`  where Tags like \"%Indie%\" and  Positive_Reviews + Negative_Reviews between 50 and 30000 and (Positive_Reviews/(Positive_Reviews + Negative_Reviews)) >=0.8"
    filtering = client.query(q1).to_dataframe()
    return origin, n_items, filtering

def get_user(api,id, origin):
    item_encoder = joblib.load('item_encoder.joblib')

    input_ = requests.get(f'https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={api}&steamid={id}&include_played_free_games=True&include_appinfo=True')
    # TODO : 인풋 부분 수정
    test = pd.DataFrame(input_.json()['response']['games'])
    test = test[['appid','name','playtime_forever']]
    test['userid'] = id # userid
    test['rating'] = 1
    test.loc[test[test['playtime_forever']<=120].index,'rating'] = 0
    test = test[['userid','appid','rating']]
    test.columns = ['userid','item_id','rating'] 
    # TODO : 이렇게 만들어진 test는 DB UIdata에 추가 팔요
        
    test = pd.merge(origin[['i']], test, left_on='i', right_on='item_id', how='inner')
    test['item_id']  = item_encoder.transform(test['item_id'])
    # FIXME : recommend를 위한 item indexing ~ indexing 과정 model 내 혹은 train/valiud 과정에 추가 필요
    
    return test , item_encoder

def get_model(model_path, n_items):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuMF(n_items, 64, 1, 0.05).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def inference(model, test, n_items, item_encoder):
    pred_list = []
    model.eval()
    
    query_user_ids = test['userid'].unique() # 추론할 모든 user array 집합
    full_item_ids = np.array([c for c in range(n_items)]) # 추론할 모든 item array 집합 
    for user_id in query_user_ids:
        with torch.no_grad():
            user_ids = np.full(n_items, user_id)
            
            user_ids = torch.LongTensor(user_ids).to('cuda')
            item_ids = torch.LongTensor(full_item_ids).to('cuda')
            
            eval_output = model.forward(user_ids, item_ids).detach().cpu().numpy()
            pred_u_score = eval_output.reshape(-1)   
        
        pred_u_idx = np.argsort(pred_u_score)[::-1]
        pred_u = full_item_ids[pred_u_idx]
        pred_list.append(list(pred_u[:100]))
        
    pred = pd.DataFrame(data=pred_list[0], columns=['App_ID'])
    pred['App_ID'] = item_encoder.inverse_transform(pred['App_ID'])
    
    return pred

def filter(pred, filtering):
    real_pred = pd.merge(pred, filtering, on='App_ID',how='inner')
    return real_pred

def main():
    start = time.time()
    userid = "userid'
    api =  "apikey"
    origin, n_items,filtering = dataload()
    test , item_encoder = get_user(api, userid, origin)
    model = get_model('../model/bestmodel_{}.pth'.format(datetime.now().day), n_items)
    pred = inference(model, test, n_items, item_encoder)
    
    real_pred = filter(pred, filtering)
    
    print(real_pred)
    print(f"{time.time()-start:.4f} sec")
    # 실제론, request부분은 선행되기에 빠르게 가능? 
    # get_user를 미리 받고, async로 정보 추가 및 inference
    
if __name__ == "__main__":
    main()
