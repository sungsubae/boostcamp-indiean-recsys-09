# 모델 trian을 위한 py파일입니다.
# train대상 data는 빅쿼리 내에 존재하고 이를 갖고와 train합니다.
# input >> 빅쿼리 데이터 // output >> Labeler, Model
# 굳이 model 서버와 inference 서버가 분리될 필요가 있을까?
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import joblib
import os
import pandas as pd

import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.init import normal_

from sklearn.preprocessing import LabelEncoder
import joblib


import warnings
warnings.filterwarnings("ignore")

## 데이터 로드

# 서비스 계정 키 JSON 파일 경로
# 사용 필요시 빅쿼리 주인 요청 필
def dataload():
    credential_path = 'key.json'
    credentials = service_account.Credentials.from_service_account_file(credential_path)
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    sql_1 = "SELECT * FROM `data.game`"
    sql_2 = "SELECT * FROM `data.interaction`"
    game = client.query(sql_1).to_dataframe()
    data = client.query(sql_2).to_dataframe()
    
    return data, game

## 데이터 프리프로세싱
def dataprocessing(data, game):
    data = pd.merge(data, game, left_on='item_id', right_on='App_ID', how='inner')
    data = data[['userid','item_id','playtime_forever']]


    data['rating'] = 1

    data.loc[data[data['playtime_forever']<=2].index,'rating'] = 0
    data.drop_duplicates(inplace=True)
    data = data[['userid','item_id','rating']]
    
    item_encoder = LabelEncoder()
    data['item_id'] = item_encoder.fit_transform(data['item_id'])
    
    user_encoder = LabelEncoder()
    data['userid'] = user_encoder.fit_transform(data['userid'])
    
    n_users = data['userid'].nunique()
    n_items = data['item_id'].nunique()
    
    joblib.dump(item_encoder, 'item_encoder.joblib')
    joblib.dump(user_encoder, 'user_encoder.joblib')
    
    matrix_rating = data.pivot_table('rating',index='userid', columns='item_id')
    matrix_rating.fillna(0, inplace=True)
    train = matrix_rating.values
    
    return train, data, game, n_users, n_items

## 모델 정의
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
    
    def forward(self, item_indices):
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
    
## 학습용 데이터셋 만들기
def make_UIdataset(train, neg_ratio):
    UIdataset = {}
    for user_id, items_by_user in enumerate(train):
        UIdataset[user_id] = []
        # positive 샘플 계산 
        pos_item_ids = np.where(items_by_user > 0.5)[0]
        num_pos_samples = len(pos_item_ids)

        # negative 샘플 계산 (random negative sampling) 
        num_neg_samples = neg_ratio * num_pos_samples
        neg_items = np.where(items_by_user < 0.5)[0]
        neg_item_ids = np.random.choice(neg_items, min(num_neg_samples, len(neg_items)), replace=False)
        UIdataset[user_id].append(np.concatenate([pos_item_ids, neg_item_ids]))
        
        # label 저장  
        pos_labels = np.ones(len(pos_item_ids))
        neg_labels = np.zeros(len(neg_item_ids))
        UIdataset[user_id].append(np.concatenate([pos_labels, neg_labels]))

    return UIdataset

def make_batchdata(UIdataset,user_indices, batch_idx, batch_size):
    batch_user_indices = user_indices[batch_idx*batch_size : (batch_idx+1)*batch_size]
    batch_user_ids = []
    batch_item_ids = []
    batch_labels = []
    for user_id in batch_user_indices:
        item_ids = UIdataset[user_id][0]
        labels = UIdataset[user_id][1]
        user_ids = np.full(len(item_ids), user_id)
        batch_user_ids.extend(user_ids.tolist())
        batch_item_ids.extend(item_ids.tolist())
        batch_labels.extend(labels.tolist())
    return batch_user_ids, batch_item_ids, batch_labels

def update_avg(curr_avg, val, idx):
    return (curr_avg * idx + val) / (idx + 1)

## 스코어링 지표
def recallk(actual, predicted, k = 25):
    set_actual = set(actual)
    recall_k = len(set_actual & set(predicted[:k])) / min(k, len(set_actual))
    return recall_k

def unique(sequence):
    # preserves order
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def ndcgk(actual, predicted, k = 25):
    set_actual = set(actual)
    idcg = sum([1.0 / np.log(i + 2) for i in range(min(k, len(set_actual)))])
    dcg = 0.0
    unique_predicted = unique(predicted[:k])
    for i, r in enumerate(unique_predicted):
        if r in set_actual:
            dcg += 1.0 / np.log(i + 2)
    ndcg_k = dcg / idcg
    return ndcg_k

def evaluation(gt, pred):
    gt = gt.groupby('userid')['item_id'].unique().to_frame().reset_index()
    gt.columns = ['profile_id', 'actual_list']

    evaluated_data = pd.merge(pred, gt, how = 'left', on = 'profile_id')

    evaluated_data['Recall@25'] = evaluated_data.apply(lambda x: recallk(x.actual_list, x.predicted_list), axis=1)
    evaluated_data['NDCG@25'] = evaluated_data.apply(lambda x: ndcgk(x.actual_list, x.predicted_list), axis=1)

    recall = evaluated_data['Recall@25'].mean()
    ndcg = evaluated_data['NDCG@25'] .mean()
    coverage = (evaluated_data['predicted_list'].apply(lambda x: x[:10]).explode().nunique())

    score = 0.75*recall + 0.25*ndcg
    rets = {"recall" :recall, 
            "ndcg" :ndcg, 
            "coverage" :coverage, 
            "score" :score}
    return rets

## train and valid

def train_epoch(UIdataset,n_users, epoch, batch_size,model, optimizer, criterion): 
    model.train()
    curr_loss_avg = 0.0

    user_indices = np.arange(n_users)
    np.random.RandomState(epoch).shuffle(user_indices)
    batch_num = int(len(user_indices) / batch_size) + 1
    bar = tqdm(range(batch_num), leave=False)
    for step, batch_idx in enumerate(bar):
        user_ids, item_ids, labels = make_batchdata(UIdataset,user_indices, batch_idx, batch_size)
        # 배치 사용자 단위로 학습
        item_ids = torch.LongTensor(item_ids).to('cuda')
        labels = torch.FloatTensor(labels).to('cuda')
        labels = labels.view(-1, 1)

        # grad 초기화
        optimizer.zero_grad()

        # 모델 forward
        output = model.forward(item_ids)
        output = output.view(-1, 1)

        loss = criterion(output, labels)

        # 역전파
        loss.backward()

        # 최적화
        optimizer.step()    
        curr_loss_avg = update_avg(curr_loss_avg, loss, step)
        
        msg = f"epoch: {epoch}, "
        msg += f"loss: {curr_loss_avg.item():.5f}, "
        msg += f"lr: {optimizer.param_groups[0]['lr']:.6f}"
        bar.set_description(msg)
    rets = {'losses': np.around(curr_loss_avg.item(), 5)}
    return rets

def valid_epoch(model, data, n_items):
    
    pred_list = []
    model.eval()
    
    query_user_ids = data['userid'].unique() # 추론할 모든 user array 집합
    full_item_ids = np.array([c for c in range(n_items)]) # 추론할 모든 item array 집합 
    for user_id in query_user_ids:
        with torch.no_grad():
            item_ids = torch.LongTensor(full_item_ids).to('cuda')
            
            
            eval_output = model.forward(item_ids).detach().cpu().numpy()
            pred_u_score = eval_output.reshape(-1)   
        
        pred_u_idx = np.argsort(pred_u_score)[::-1]
        pred_u = full_item_ids[pred_u_idx]
        pred_list.append(list(pred_u[:50]))
        
    pred = pd.DataFrame()
    pred['profile_id'] = query_user_ids
    pred['predicted_list'] = pred_list
    
    # 모델 성능 확인 
    rets = evaluation(data, pred)
    return rets, pred

def training(train,data ,n_users, n_items, epochs, batch_size):
    model = NeuMF(n_items, 64, 1,0.05).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    best_scores  = 0
    for epoch in range(epochs):
        train_results = train_epoch(train,n_users,epoch,batch_size,model, optimizer, criterion)
        
        # cfg.check_epoch 번의 epoch 마다 성능 확인 
        
        valid_results, _ = valid_epoch(model, data, n_items)
        # 검증 성능 확인 
        print(epoch,' : recall >> ' ,valid_results['score'])
                    
        # 가장 성능이 좋은 가중치 파일을 저장 
        if best_scores <= valid_results['score']: 
            best_scores = valid_results['score']
            torch.save(model.state_dict(), '../model/bestmodel_{}.pth'.format(datetime.now().day))
    
    
def main():
    data, game = dataload()
    train, data, game, n_users, n_items = dataprocessing(data, game)
    matrix = make_UIdataset(train, 2)
    training(matrix, data ,n_users, n_items, epochs=1, batch_size=1024)
    
if __name__ =="__main__":
    main()