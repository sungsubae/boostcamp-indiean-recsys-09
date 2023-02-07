"""
https://github.com/Darel13712/ease_rec
"""

from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count

import pickle


class EASE:
    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, 'user_id'])
        items = self.item_enc.fit_transform(df.loc[:, 'item_id'])
        return users, items

    def fit(self, df, lambda_: float = 250, implicit=True):
        """
        df: pandas.DataFrame with columns user_id, item_id and (rating)
        lambda_: l2-regularization term
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        """
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
        dd = train.loc[train.user_id.isin(users)]
        dd['ci'] = self.item_enc.transform(dd.item_id)
        dd['cu'] = self.user_enc.transform(dd.user_id)
        g = dd.groupby('cu')
        with Pool(cpu_count()) as p:
            user_preds = p.starmap(
                self.predict_for_user,
                [(user, group, self.pred[user, :], items, k) for user, group in g],
            )
        df = pd.concat(user_preds)
        df['item_id'] = self.item_enc.inverse_transform(df['item_id'])
        df['user_id'] = self.user_enc.inverse_transform(df['user_id'])
        return df

    @staticmethod
    def predict_for_user(user, group, pred, items, k):
        watched = set(group['ci'])
        candidates = [item for item in items if item not in watched]
        pred = np.take(pred, candidates)
        res = np.argpartition(pred, -k)[-k:]
        r = pd.DataFrame(
            {
                "user_id": [user] * len(res),
                "item_id": np.take(candidates, res),
                "score": np.take(pred, res),
            }
        ).sort_values('score', ascending=False)
        return r

    #################### By KCH ####################

    # 밸리드 셋에 대한 오프라인 성능 확인, Recall@K => 상호작용 데이터를 전부 넣고 맞췄는지는 데이터 리키지인 것 같다... 마지막 10개를 뺀 나머지 상호작용 데이터만 넣고 검증 진행하자!
    def evaluation(self, valid_df, k=10):
        recalls = 0
        cnt = 0
        for u in valid_df['user_id'].unique():
            data = valid_df[valid_df['user_id'] == u]
            if len(data) > k:
                target = data.tail(k)
                input = data.head(len(data) - k)
                value = np.ones(input.shape[0])
                user = np.zeros(input.shape[0])
                try:
                    item = self.item_enc.transform(input.loc[:, 'item_id'].unique())
                except:
                    continue

                x = csr_matrix((value, (user, item)), shape=(1, self.B.shape[0]))
                pred = x.dot(self.B)[0]

                res = np.argpartition(pred, -k)[-k:]
                try:
                    recall = len(set(res) & set(self.item_enc.transform(target.loc[:, 'item_id'].unique()))) / k
                    cnt += 1
                except:
                    continue

                recalls += recall
        recalls = recalls / cnt
        print(f"Recall@{k}: {recalls:.4f} for {cnt} users")

        return recalls

    # 행렬 B를 활용하여 인퍼런스 - 유저 프리
    def inference(self, target_df, k=10):
        if not (set(target_df['item_id'].unique()) & set(self.item_enc.classes_)):
            print("아이템 콜드스타트 문제로 인해 인퍼런스가 불가능합니다!")
            return
        target = list(set(target_df['item_id'].unique()) & set(self.item_enc.classes_))
        value = np.ones(len(target))
        user = np.zeros(len(target))
        item = self.item_enc.transform(target)

        x = csr_matrix((value, (user, item)), shape=(1, self.B.shape[0]))
        pred = x.dot(self.B)[0]

        pred[item] = -10
        res = np.argpartition(pred, -k)[-k:]
        filtered_r = pd.DataFrame(
            {
                "user_id": [target_df['user_id'].unique()] * len(res),
                "item_id": res,
                "score": np.take(pred, res),
            }
        ).sort_values('score', ascending=False)
        filtered_r['item_id'] = self.item_enc.inverse_transform(filtered_r['item_id'])

        return filtered_r

    # 학습한 모델 저장
    def save(self, filename='model.sav'):
        pickle.dump(self, open(filename, 'wb'))

    # 학습한 모델 불러오기 => 약 3초 소요
    def load(self, filename='model.sav'):
        return pickle.load(open(filename, 'rb'))

    # 학습한 모델 저장 for 인퍼런스
    def save_for_inference(self, filename='data_for_inference.sav'):
        data = {
            'item_enc': self.item_enc,
            'B': self.B
        }
        pickle.dump(data, open(filename, 'wb'))

    # 학습한 모델 불러오기 for 인퍼런스 => 약 0.5초 소요
    def load_for_inference(self, filename='data_for_inference.sav'):
        data = pickle.load(open(filename, 'rb'))
        self.item_enc = data['item_enc']
        self.B = data['B']