import datetime
import os
import random
from typing import Tuple
from zipfile import ZipFile

import imageio
import winsound
from matplotlib import pyplot as plt

from app.config import settings


class GtzanDataset:
    def __init__(self):
        self.gtzan_dataset_zip_path: str = settings.gtzan_zip_path
        self.gtzan_genres_original_path: str = settings.gtzan_genres_original
        self.gtzan_images_original_path: str = settings.gtzan_images_original

    def list_files_info(self) -> None:
        # opening the zip file in READ mode
        with ZipFile(self.gtzan_dataset_zip_path, "r") as zip_file:
            for info in zip_file.infolist():
                print(info.filename)
                print("\tModified:\t" + str(datetime.datetime(*info.date_time)))
                print("\tSystem:\t\t" + str(info.create_system) + "(0 = Windows, 3 = Unix)")
                print("\tZIP version:\t" + str(info.create_version))
                print("\tCompressed:\t" + str(info.compress_size) + " bytes")
                print("\tUncompressed:\t" + str(info.file_size) + " bytes")

    def extract(self) -> None:
        with ZipFile(self.gtzan_dataset_zip_path, 'r') as zipObj:
            # Extract all the contents of zip file in different directory
            path = './data/gztan/data'
            zipObj.extractall(path)

    def _get_random_original_file(self) -> str:
        directory_name = random.choice(os.listdir(self.gtzan_genres_original_path))
        file_name = random.choice(os.listdir(self.gtzan_genres_original_path + "\\" + directory_name))
        file_path = f"{self.gtzan_genres_original_path}\\{directory_name}\\{file_name}"
        print(f"Random original file path: \n\t{file_path}")
        return file_path

    def _get_random_spectogram_file(self) -> Tuple[str, str]:
        directory_name = random.choice(os.listdir(self.gtzan_images_original_path))
        file_name = random.choice(os.listdir(self.gtzan_images_original_path + "\\" + directory_name))
        file_path = f"{self.gtzan_images_original_path}\\{directory_name}\\{file_name}"
        print(f"Random spectogram file path: \n\t{file_path}")
        return file_path, file_name

    def play_random_original_file(self) -> None:
        file_path = self._get_random_original_file()
        winsound.PlaySound(file_path, winsound.SND_FILENAME)

    def show_random_waveplot(self) -> None:
        file_path, file_name = self._get_random_spectogram_file()

        img = imageio.imread(file_path)
        print(f"Image dimensions: \n\t{img.shape}")

        plt.imshow(img, interpolation='nearest')
        plt.title(file_name)
        plt.show()

