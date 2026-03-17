import os
import pandas as pd
from typing import List
from datetime import datetime
from fastapi import FastAPI
from sqlalchemy import create_engine
from catboost import CatBoostClassifier
from pydantic import BaseModel
import hashlib

# --- SCHEMA ---
class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]

# --- CONFIG ---
class Config:
    DB_CONN = os.getenv(
        "DB_CONN",
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
    )
    # Имя файла модели. В LMS оно должно лежать рядом с app.py
    MODEL_NAME = "model_test.cbm"

    TABLE_USERS = os.getenv("TABLE_USERS", "public.user_data")
    TABLE_POSTS = os.getenv("TABLE_POSTS", "valeriya_nikitina_posts_lesson_22")
    TABLE_LIKES = os.getenv("TABLE_LIKES", "public.feed_data")
    TABLE_SIMILARITY = os.getenv("TABLE_SIMILARITY", "valeriya_nikitina_score_lesson_22")


params = Config()

app = FastAPI()


# --- LOADERS ---

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(params.DB_CONN)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        for col in chunk_dataframe.select_dtypes(include=['float64']).columns:
            chunk_dataframe[col] = chunk_dataframe[col].astype('float32')
        for col in chunk_dataframe.select_dtypes(include=['int64']).columns:
            chunk_dataframe[col] = chunk_dataframe[col].astype('int32')
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_user() -> pd.DataFrame:
    query = f"SELECT user_id, gender, age, country, city, exp_group, os FROM {params.TABLE_USERS}"
    return batch_load_sql(query)


def load_posts() -> pd.DataFrame:
    return batch_load_sql(f'SELECT * FROM {params.TABLE_POSTS}')


def load_item() -> dict:
    try:
        query = f"SELECT post_id, neighbor_id, score FROM {params.TABLE_SIMILARITY}"
        df = batch_load_sql(query)
        sim_dict = {}
        for pid, nid, score in zip(df['post_id'], df['neighbor_id'], df['score']):
            if pid not in sim_dict:
                sim_dict[pid] = {}
            sim_dict[pid][nid] = score
        del df
        return sim_dict
    except Exception as e:
        print(f"Error loading similarity: {e}")
        return {}


def load_user_likes() -> dict:
    # Загружаем ТОЛЬКО лайки для экономии памяти
    query = f"SELECT user_id, post_id FROM {params.TABLE_LIKES} WHERE action='like'"
    df = batch_load_sql(query)
    likes_dict = df.groupby('user_id')['post_id'].apply(list).to_dict()
    del df
    return likes_dict



def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = f'/workdir/user_input/{path}'
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(current_dir, '..', 'model', f'{path}.cbm')
    return MODEL_PATH

def load_models():
    model_control = CatBoostClassifier()
    path_control = get_model_path("model_control")
    model_control.load_model(path_control)

    model_test = CatBoostClassifier()
    path_test = get_model_path("model_test")
    model_test.load_model(path_test)

    return model_control, model_test

# --- GLOBAL VARIABLES ---
print("Loading data...")
users = load_user()
posts = load_posts()
item = load_item()
likes = load_user_likes()
model_control, model_test = load_models()
print("Data loaded successfully.")


# --- LOGIC ---

SALT = 'str_for_salt'

def get_exp_group(user_id: int) -> str:
    val_str = str(user_id) + SALT

    hash_val = int(hashlib.md5(val_str.encode()).hexdigest(), 16)

    if hash_val % 2 ==0:
        return 'control'
    else:
        return 'test'

def calc_sim_score(user_likes, candid_post_id):
    if candid_post_id not in item:
        return 0.0
    score = 0.0
    similar_posts = item[candid_post_id]
    for liked_post in user_likes:
        if liked_post in similar_posts:
            score += similar_posts[liked_post]
    return score


def get_recommended_posts_control(user_id: int, time: datetime, limit: int):
    user_features = users[users['user_id'] == user_id]
    if user_features.empty:
        return posts['post_id'].head(limit).tolist()

    user_likes_list = likes.get(user_id, [])
    candidates = posts.copy()

    u_feat_vals = user_features.iloc[0].to_dict()
    for col, val in u_feat_vals.items():
        candidates[col] = val

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

    probs = model_control.predict_proba(df_for_predict)[:, 1]
    candidates['proba'] = probs

    top_posts = candidates.sort_values('proba', ascending=False).head(limit)
    return top_posts['post_id'].tolist()

def get_recommended_posts_test(user_id: int, time: datetime, limit: int):
    user_features = users[users['user_id'] == user_id]
    if user_features.empty:
        return posts['post_id'].head(limit).tolist()

    user_likes_list = likes.get(user_id, [])
    candidates = posts.copy()

    u_feat_vals = user_features.iloc[0].to_dict()
    for col, val in u_feat_vals.items():
        candidates[col] = val

    candidates['month'] = time.month
    candidates['hour'] = time.hour

    candidates['item_similarity_score'] = candidates['post_id'].map(
        lambda pid: calc_sim_score(user_likes_list, pid)
    )

    n_comp = 15
    pca_columns = [f'text_pca_{i}' for i in range(n_comp)]
    train_columns = [
                        'gender', 'age', 'country', 'city', 'exp_group', 'os',
                        'topic', 'item_similarity_score', 'month', 'hour'
                    ] + pca_columns

    missing = [c for c in train_columns if c not in candidates.columns]
    if missing:
        for c in missing: candidates[c] = 0


    df_for_predict = candidates[train_columns]

    probs = model_test.predict_proba(df_for_predict)[:, 1]
    candidates['proba'] = probs

    top_posts = candidates.sort_values('proba', ascending=False).head(limit)
    return top_posts['post_id'].tolist()


@app.get('/post/recommendations/', response_model=Response)
def recommended_post(id: int, time: datetime, limit: int = 5) -> Response:

    exp_group = get_exp_group(id)

    if exp_group == 'control':
        post_ids = get_recommended_posts_control(id, time, limit)
    elif exp_group == 'test':
        post_ids = get_recommended_posts_test(id, time, limit)
    else:
        raise ValueError('Unknown group')

    res = []
    for pid in post_ids:
        post_info = posts[posts['post_id'] == pid]
        if not post_info.empty:
            rec = post_info.iloc[0]
            txt = rec['text'] if 'text' in rec else "No text content"
            res.append(PostGet(
                id=pid,
                text=str(txt),
                topic=rec['topic']
            ))
        else:
            res.append(PostGet(id=pid, text="Error", topic="Error"))

    return Response(exp_group = exp_group,recommendations = res)