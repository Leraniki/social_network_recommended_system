import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DB_CONN = os.getenv('DB_CONN')

    MODEL_PATH = os.getenv('MODEL_PATH')

    TABLE_USERS = os.getenv('TABLE_USERS')
    TABLE_POSTS = os.getenv('TABLE_POSTS')
    TABLE_LIKES = os.getenv('TABLE_LIKES')
    TABLE_SIMILARITY = os.getenv('TABLE_SIMILARITY')

params = Config()