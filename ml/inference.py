import os
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.init import normal_
import requests
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import joblib

import warnings
warnings.filterwarnings("ignore")

class NeuMF(nn.Module):
    def __init__(self, n_items, emb_dim, layer_dim,dropout):

        super(NeuMF, self).__init__()
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.layer_dim = layer_dim
        self.dropout = dropout
        self.build_graph()

    def build_graph(self):
        #self.user_embedding_mf = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.emb_dim)
        self.item_embedding_mf = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.emb_dim)
        
        #self.user_embedding_mlp = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.emb_dim)
        self.item_embedding_mlp = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.emb_dim)
                
        
        self.mlp_layers = nn.Sequential(
            nn.Linear(self.emb_dim, self.layer_dim), 
            nn.ReLU(), 
            nn.Dropout(p=self.dropout), 
            nn.Linear(self.layer_dim, self.layer_dim//2), 
            nn.ReLU(), 
            nn.Dropout(p=self.dropout)
        )
        self.affine_output = nn.Linear(self.layer_dim//2 + self.emb_dim, 1)
        self.apply(self._init_weights)
        

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)
        elif isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    def forward(self, user_indices, item_indices):
        #user_embedding_mf = self.user_embedding_mf(user_indices)
        item_embedding_mf = self.item_embedding_mf(item_indices)
        mf_output = item_embedding_mf
        
        #user_embedding_mlp = self.user_embedding_mlp(user_indices)
        item_embedding_mlp = self.item_embedding_mlp(item_indices)
        input_feature = torch.cat([item_embedding_mlp[0]], -1)
        mlp_output = self.mlp_layers(input_feature)
        
        output = torch.cat([mlp_output, mf_output], dim=-1)
        output = self.affine_output(output).squeeze(-1)
        return output

def inference_epoch(model, test, n_items, item_encoder):
    
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
        pred_list.append(list(pred_u[:50]))
        
    pred = pd.DataFrame()
    pred['profile_id'] = query_user_ids
    pred['predicted_list'] = pred_list

    return pred

def get_user(id, origin):
    item_encoder = joblib.load('item_encoder.joblib')

    input_ = requests.get(f'https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key=F644C6E2851DBDC938FF6FC70388451F&steamid={id}&include_played_free_games=True&include_appinfo=True')
    # TODO : 인풋 부분 수정
    test = pd.DataFrame(input_.json()['response']['games'])
    test = test[['appid','name','playtime_forever']]
    test['userid'] = id # userid
    test['rating'] = 1
    test.loc[test[test['playtime_forever']<=120].index,'rating'] = 0
    test = test[['userid','appid','rating']]
    test.columns = ['userid','item_id','rating'] 
    # TODO : 이렇게 만들어진 test는 DB UIdata에 추가 팔요
        
    test = pd.merge(origin[['item_id']].drop_duplicates(), test, left_on='item_id', right_on='item_id', how='inner')
    # FIXME : UIdata에 없는 item drop하는 부분 수정 필요
    test['item_id']  = item_encoder.transform(test['item_id'])
    # FIXME : recommend를 위한 item indexing ~ indexing 과정 model 내 혹은 train/valiud 과정에 추가 필요
    
    return test 

def inference(model, test, n_items):
    item_encoder = joblib.load('item_encoder.joblib')

    model = model(n_items, 64, 1,0.05).to('cuda')
    model.load_state_dict(torch.load('best_model.pth'))
    # TODO : 모델 파일 수정

    pred_result = inference_epoch(model, test, n_items, item_encoder)
    for i in pred_result.predicted_list:

        real_pred = item_encoder.inverse_transform(i)

    return real_pred 


def main():
    origin = pd.read_csv('useritem.csv')
    test_input = get_user(76561198117856251, origin)
    pred_list = inference_(NeuMF, test_input, 8735)
    
    return pred_list
    
if __name__ == "__main__":
    main()
    
    
##########################################################################################

# Load Model
def get_model(model_path: str = "ml/best_model.pth") -> NeuMF:
    """Model을 가져옵니다"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuMF(8735, 64, 1,0.05).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model
    
# get recommendation result
def get_model_rec_prototype(get_user, model, inference):
    origin = pd.read_csv('useritem.csv')
    test_input = get_user(76561198117856251, origin)
    pred_list = inference(model, test_input, 8735)
    
    return pred_list

def inference_(model, test, n_items):
    item_encoder = joblib.load('item_encoder.joblib')

    model = model(n_items, 64, 1,0.05).to('cuda')
    model.load_state_dict(torch.load('best_model.pth'))
    # TODO : 모델 파일 수정

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
        pred_list.append(list(pred_u[:50]))
    
    ######################################################################################################
    # 코드 수정 필요 
    df = pd.DataFrame()
    df['profile_id'] = query_user_ids
    df['predicted_list'] = pred_list
    # 어짜피 User 한 명씩 inference하니까 반복문 밖에서 코드 정의
    
    titles = {}
    posters = {}
    # top_k : 10 , 가져온 index를 기반으로 df상에서 title, image 가져와야 됌
    for col_index in range(
                10
            ):
            steam = df.iloc[col_index]
            titles[col_index] = steam["title"]
            posters[col_index] = steam["poster_link"] if steam["poster_link"] else Image.open("placeholder.png")
    
    title = [str(x) for x in titles.values()]
    poster = [str(x) for x in posters.values()]
    
    return title, poster 