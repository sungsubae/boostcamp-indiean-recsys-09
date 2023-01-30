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
        items = self.item_enc.transform(items)
        dd = train.loc[train.userid.isin(users)]
        dd['ci'] = self.item_enc.transform(dd.item_id)
        dd['cu'] = self.user_enc.transform(dd.userid)
        g = dd.groupby('cu')
        
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
        watched = set(group['ci'])
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
    credential_path = 'key.json'
    credentials = service_account.Credentials.from_service_account_file(credential_path)
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    q="SELECT DISTINCT App_ID, Genre, Positive_Reviews, Negative_Reviews FROM `data.game`"
    game = client.query(q).to_dataframe()
    
    q="SELECT DISTINCT item_id, userid, playtime_forever FROM `data.new_interaction`"
    data = client.query(q).to_dataframe()
    
    return data, game

    
def get_user(userid,api):
    input_ = requests.get(f'https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={api}&steamid={userid}&include_played_free_games=True&include_appinfo=True')
    test = pd.DataFrame(input_.json()['response']['games'])
    test = test[['appid','playtime_forever']]
    test['userid'] = userid
    test.columns = ['item_id','playtime_forever' ,'userid']
    return test

def train_predict(train, test, game): 
    model = EASE()
    train = pd.concat([train, test]).reset_index()
    train['rating'] = 1
    train.loc[train[train['playtime_forever']<=120].index,'rating'] = 0
    model.fit(train, 0.5, implicit=False)
    output = model.predict(test, test['userid'].unique(), train['item_id'].unique(), 50)
    list_ = game[(game['Genre'].str.contains('Indie', na=False)) & (game['Positive_Reviews']>=game['Negative_Reviews']*4) & (game['Positive_Reviews']+game['Negative_Reviews']<=50000)]['App_ID'].values
    output = output[output['item_id'].isin(list_)]['item_id'].values
    return output

def main():
    userid=76561198117856251
    train, game = dataload()
    test = get_user(userid) 
    output = train_predict(train, test, game)
    return output

if __name__ == "__main__":
    main()