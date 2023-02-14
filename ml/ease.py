from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
from google.cloud import bigquery
from google.oauth2 import service_account
import requests
import time

class EASE:
    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, 'userid'])
        items = self.item_enc.fit_transform(df.loc[:, 'item_id'])
        return users, items

    def fit(self, df, lambda_: float = 0.5, implicit=True):
        users, items = self._get_users_and_items(df)
        values = (
            np.ones(df.shape[0])
            if implicit
            else df['rating'].to_numpy() / df['rating'].max()
        )

        X = csr_matrix((values, (users, items)))
        self.X = X

        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B
        self.pred = X.dot(B)
        
        
    def predict(self, train, users, items, k):
        items = self.item_enc.fit_transform(items)
        dd = train.loc[train.userid.isin(users)]
        print(dd)
        print(type(dd.item_id))
        print(dd.item_id)
        dd['item_id'] = self.item_enc.transform(dd.item_id.astype(str))
        dd['userid'] = self.user_enc.transform(dd.userid)
        g = dd.groupby('userid')
        
        with Pool(cpu_count()) as p:
            user_preds = p.starmap(
                self.predict_for_user,
                [(user, group, self.pred[user, :], items, k) for user, group in g],
            )
        df = pd.concat(user_preds)
        df['item_id'] = self.item_enc.inverse_transform(df['item_id'])
        df['userid'] = self.user_enc.inverse_transform(df['userid'])
        
        return df
    
    def evaluation(train, test): 
        model = EASE()
        train = pd.concat([train, test]).reset_index()
        train['rating'] = 1
        train.loc[train[train['playtime_forever']<=120].index,'rating'] = 0
        model.fit(train, 0.5, implicit=False)
        output = model.evaluate(train, train['userid'].unique(), train['item_id'].unique(), 10)
        evale = pd.merge(train[train['rating']==1], output, on=['userid','item_id'], how='inner')
        precision10 = (evale.groupby('userid').count()['index']/10).values.mean()
        return precision10
    
    @staticmethod
    def predict_for_user(user, group, pred, items, k):
        watched = set(group['item_id'])
        candidates = [item for item in items if item not in watched]
        pred = np.take(pred, candidates)
        res = np.argpartition(pred, -k)[-k:]
        r = pd.DataFrame(
            {
                "userid": [user] * len(res),
                "item_id": np.take(candidates, res),
                "score": np.take(pred, res),
            }
        ).sort_values('score', ascending=False)
        
        return r
    
def dataload():
    credential_path = '/opt/ml/level3_Final_project/final-project-level3-recsys-09/glassy-droplet-375219-9e50b4fc0381.json'
    credentials = service_account.Credentials.from_service_account_file(credential_path)
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    q="SELECT DISTINCT App_ID, Genre, Positive_Reviews, Negative_Reviews FROM `data.game`"
    game = client.query(q).to_dataframe()
    
    q="SELECT DISTINCT item_id, userid, playtime_forever FROM `data.new_interaction`"
    data = client.query(q).to_dataframe()
    
    return data, game

    
def get_user(userid, playtime_forever, gameid_list):
    data = {'userid': [str(userid)] * len(gameid_list), 'playtime_forever': playtime_forever, 'item_id' : [str(x) for x in gameid_list]}
    test = pd.DataFrame(data)
    return test

def inference(train, test, game, model): 
    train.item_id = train.item_id.astype(str)
    train.userid = train.userid.astype(str)
    train = pd.concat([train, test]).reset_index()
    train['rating'] = 1
    train.loc[train[train['playtime_forever']<=120].index,'rating'] = 0
    
    model.fit(train, 0.5, implicit=False)
    output = model.predict(test, test['userid'].unique(), train['item_id'].unique(), 500)
    list_ = game[(game['Genre'].str.contains('Indie', na=False))]['App_ID'].values.astype(str)
    output = output[output['item_id'].isin(list_)]['item_id'].values
    
    return output.tolist()

# def main():
#     userid=76561198117856251
#     train, game = dataload()
#     test = get_user(userid, api) 
#     output = inference(train, test, game)
#     return output

# if __name__ == "__main__":
#     main()

def get_model():
    return EASE
