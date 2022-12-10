from typing import Optional

from pydantic import BaseSettings


class Settings(BaseSettings):
    gtzan_zip_path: Optional[str]
    gtzan_genres_original: Optional[str]
    gtzan_genres_3_sec_original: Optional[str]
    gtzan_images_original: Optional[str]
    gtzan_images_3_sec_original: Optional[str]

    gztan_train_dir: Optional[str]
    gztan_validation_dir: Optional[str]
    gztan_test_dir: Optional[str]

    gztan_train_3sec_dir: Optional[str]
    gztan_validation_3sec_dir: Optional[str]
    gztan_test_3sec_dir: Optional[str]

    fma_metadata_path: Optional[str]
    fma_genres_path: Optional[str]
    fma_small_dataset_path: Optional[str]
    fma_images_path: Optional[str]

    fma_train_dir: Optional[str]
    fma_validation_dir: Optional[str]
    fma_test_dir: Optional[str]

    class Config:
        env_file = "../ENV/local.env"
        env_file_encoding = "utf-8"


settings = Settings()
