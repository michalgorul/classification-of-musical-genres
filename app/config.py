from pydantic import BaseSettings


class Settings(BaseSettings):
    gtzan_zip_path: str
    gtzan_genres_original: str
    gtzan_genres_3_sec_original: str
    gtzan_images_original: str
    gtzan_images_3_sec_original: str

    gztan_train_dir: str
    gztan_validation_dir: str
    gztan_test_dir: str

    gztan_train_3sec_dir: str
    gztan_validation_3sec_dir: str
    gztan_test_3sec_dir: str

    fma_metadata_path: str
    fma_genres_path: str
    fma_small_dataset_path: str
    fma_images_path: str

    fma_train_dir: str
    fma_validation_dir: str
    fma_test_dir: str

    class Config:
        env_file = "../ENV/local.env"
        env_file_encoding = "utf-8"


settings = Settings()
