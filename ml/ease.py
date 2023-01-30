from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
from google.cloud import bigquery
from google.oauth2 import service_account
import requests

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
    
    
    @staticmethod
    def predict_for_user(user, group, pred, items, k):
        watched = set(group['ci'])
        candidates = [item for item in items]
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
    credential_path = '/opt/ml/input/final_project/ml/key.json'
    credentials = service_account.Credentials.from_service_account_file(credential_path)
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    q="SELECT DISTINCT * FROM `data.game`"
    game = client.query(q).to_dataframe()
    
    q="SELECT DISTINCT * FROM `data.interaction`"
    data = client.query(q).to_dataframe()
    
    data = data[['userid','item_id','playtime_forever']]
    c = data.groupby('userid').count().sort_values(by='item_id')
    
    data_20 = c[(c['item_id']<=20)].reset_index()
    data_20 = data[data['userid'].isin(data_20['userid'])]
    
    return data_20, game

    
def get_user(userid):
    input_ = requests.get(f'https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key=F644C6E2851DBDC938FF6FC70388451F&steamid={userid}&include_played_free_games=True&include_appinfo=True')
    test = pd.DataFrame(input_.json()['response']['games'])
    test = test[['appid','playtime_forever']]
    test['userid'] = 'JWS'
    test.columns = ['item_id','playtime_forever' ,'userid']
    return test


def train_predict(train, test, game): 
    model = EASE()
    train = pd.concat([train, test]).reset_index()
    train['rating'] = 1
    train.loc[train[train['playtime_forever']<=120].index,'rating'] = 0
    model.fit(train, 0.5, implicit=False)
    output = model.predict(test, test['userid'].unique(), train['item_id'].unique(), 50)
    output = pd.merge(game, output, left_on='App_ID', right_on='item_id', how='inner')
    output = output[(output['Genre'].str.contains('Indie', na=False)) & (output['Positive_Reviews']>=output['Negative_Reviews']*4) & (output['Positive_Reviews']+output['Negative_Reviews']<=50000)][['App_ID', 'Name']]
    
    return output['App_ID'].values

def main(userid=76561198117856251):
    train, game = dataload()
    test = get_user(userid) 
    output = train_predict(train, test, game)
    
    return output

if __name__ == "__main__":
    main()
    