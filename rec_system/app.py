from fastapi import FastAPI
from typing import List
from schema import PostGet
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine

import os
from catboost import CatBoostClassifier

from config import params

db_conn = params.DB_CONN

engine = create_engine(params.DB_CONN)

app = FastAPI()

# Загрузка данных
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(db_conn)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_user() -> pd.DataFrame:
    return batch_load_sql(f'SELECT * FROM {params.TABLE_USERS}')


def load_posts() -> pd.DataFrame:
    return batch_load_sql(f'SELECT * FROM {params.TABLE_POSTS}')


def load_item() -> pd.DataFrame:
    df = batch_load_sql(f'SELECT * FROM {params.TABLE_SIMILARITY}')

    sim_dict = {}
    for pid, nid, score in zip(df['post_id'], df['neighbor_id'], df['score']):
        if pid not in sim_dict:
            sim_dict[pid] = {}
        sim_dict[pid][nid] = score

    return sim_dict


def load_user_likes():
    df = batch_load_sql(f'SELECT * FROM {params.TABLE_LIKES}')
    likes_dict = df.groupby('user_id')['post_id'].apply(list).to_dict()

    return likes_dict


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        model_path = params.MODEL_PATH
    else:
        model_path = path
    return model_path


def load_models():
    model = CatBoostClassifier()
    model_path = get_model_path("model/catboost_model_5.cbm")
    model.load_model(model_path)

    return model


users = load_user()
posts = load_posts()
model = load_models()
item = load_item()
likes = load_user_likes()


# Функции для реализации подхода item-based
def calc_sim_score(user_likes, candid_post_id):
    if candid_post_id not in item:
        return 0

    score = 0
    similar_posts = item[candid_post_id]

    for liked_post in user_likes:
        if liked_post in similar_posts:
            score += similar_posts[liked_post]
    return score


def get_recommended_posts(user_id, time, limit):
    user_features = users[users['user_id'] == user_id]

    if user_features.empty:
        return []

    user_likes_list = likes.get(user_id, [])

    candidates = posts.copy()
    candidates = candidates.assign(**user_features.iloc[0])

    candidates['month'] = time.month
    candidates['hour'] = time.hour

    candidates['item_similarity_score'] = candidates['post_id'].map(
        lambda pid: calc_sim_score(user_likes_list, pid)
    )

    train_columns = [
        'gender', 'age', 'country', 'city', 'exp_group', 'os',
        'topic', 'item_similarity_score', 'month', 'hour'
    ]

    df_for_predict = candidates[train_columns]

    probs = model.predict_proba(df_for_predict)[:, 1]
    candidates['proba'] = probs

    top_posts = candidates.sort_values('proba', ascending=False).head(limit)

    return top_posts['post_id'].tolist()


# Эндпоинт
@app.get('/post/recommendations/', response_model=List[PostGet])
def recommended_post(
        id: int,
        time: datetime,
        limit: int = 5) -> List[PostGet]:
    post_ids = get_recommended_posts(id, time, limit)

    res = []
    for pid in post_ids:
        post_info = posts[posts['post_id'] == pid].iloc[0]
        res.append(PostGet(
            id=pid,
            text=post_info['text'],
            topic=post_info['topic']
        ))
    return res
