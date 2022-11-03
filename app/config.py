from pydantic import BaseSettings


class Settings(BaseSettings):
    gtzan_zip_path: str
    gtzan_genres_original: str
    gtzan_images_original: str

    class Config:
        env_file = "../ENV/local.env"
        env_file_encoding = "utf-8"


settings = Settings()
