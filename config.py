import os
from pydantic_settings import BaseSettings


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class Settings(BaseSettings):
    APPLICATION_NAME: str = "Multiple Object Tracking"
    IMAGE_FOLDER: str
    OUT_FOLDER: str

    class Config:
        case_sensitive = True
        env_file_encoding = "utf-8"
        env_file = os.path.join(ROOT_DIR, '.env')


class Params(BaseSettings):
    YOLO_MODEL: str = 'yolov8n.pt'


settings = Settings()
params = Params()
