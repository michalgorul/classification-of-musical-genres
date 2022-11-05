import datetime
import os
import random
from typing import Tuple
from zipfile import ZipFile

import imageio
import librosa
import librosa.display
import numpy as np
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
        with ZipFile(self.gtzan_dataset_zip_path, "r") as zipObj:
            # Extract all the contents of zip file in different directory
            path = "./data/gztan/data"
            zipObj.extractall(path)

    def _get_random_original_file(self) -> Tuple[str, str]:
        directory_name = random.choice(os.listdir(self.gtzan_genres_original_path))
        file_name = random.choice(
            os.listdir(self.gtzan_genres_original_path + "\\" + directory_name)
        )
        file_path = f"{self.gtzan_genres_original_path}\\{directory_name}\\{file_name}"
        print(f"Random original file path: \n\t{file_path}\n")
        return file_path, file_name

    def _get_random_spectogram_file(self) -> Tuple[str, str]:
        directory_name = random.choice(os.listdir(self.gtzan_images_original_path))
        file_name = random.choice(
            os.listdir(self.gtzan_images_original_path + "\\" + directory_name)
        )
        file_path = f"{self.gtzan_images_original_path}\\{directory_name}\\{file_name}"
        print(f"Random spectogram file path: \n\t{file_path}\n")
        return file_path, file_name

    def _get_original_file_path_and_name(self, file_path: str | None) -> Tuple[str, str]:
        if not file_path:
            return self._get_random_original_file()
        return file_path, file_path.split("\\")[-1]

    def _get_spectogram_file_path_and_name(self, file_path: str | None) -> Tuple[str, str]:
        if not file_path:
            return self._get_random_spectogram_file()
        return file_path, file_path.split("\\")[-1]

    def play_original_file(self, original_file_path: str | None) -> None:
        try:
            file_path, _ = self._get_original_file_path_and_name(file_path=original_file_path)
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
        except Exception as e:
            print(f"Failed to play original file, err={e}")
        return

    def show_sound_wave(self, original_file_path: str | None) -> None:
        wav_file_path, wav_file_name = self._get_original_file_path_and_name(
            file_path=original_file_path
        )
        print(wav_file_name)

        y, sample_rate = librosa.load(wav_file_path)

        print("y:", y, "\n")
        print("y shape:", np.shape(y), "\n")
        print("Sample rate (KHz):", sample_rate, "\n")
        print(f"Length of audio: {np.shape(y)[0] / sample_rate} seconds")

        # Plot th sound wave.

        plt.figure(figsize=(15, 5))
        librosa.display.waveshow(y=y, sr=sample_rate)
        plt.title(f"Sound wave of {wav_file_name}", fontsize=20)
        plt.show()
        return

    def show_spectogram_from_dataset(self, image_file_path: str | None) -> None:
        file_path, file_name = self._get_spectogram_file_path_and_name(file_path=image_file_path)

        img = imageio.v2.imread(file_path)
        print(f"Image dimensions: \n\t{img.shape}")

        plt.imshow(img)
        plt.title(file_name)
        plt.show()
        return

    def create_decibel_spectogram_from_sound_file(self, sound_file_path: str | None) -> None:
        wav_file_path, wav_file_name = self._get_original_file_path_and_name(
            file_path=sound_file_path
        )
        y, sample_rate = librosa.load(wav_file_path)

        # Short-time Fourier transform (STFT).
        d = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        print("Shape of d object:", np.shape(d))
        # Convert amplitude spectrogram to Decibels-scaled spectrogram.
        decibels = librosa.amplitude_to_db(d, ref=np.max)
        # Creating the spectogram.
        plt.figure(figsize=(16, 6))
        librosa.display.specshow(
            decibels, sr=sample_rate, hop_length=512, x_axis="time", y_axis="log"
        )
        plt.colorbar()
        plt.title("Decibels-scaled spectrogram", fontsize=20)
        plt.show()
        return

    def create_mel_spectogram_from_sound_file(self, sound_file_path: str | None) -> None:
        wav_file_path, wav_file_name = self._get_original_file_path_and_name(
            file_path=sound_file_path
        )
        y, sample_rate = librosa.load(wav_file_path)

        s = librosa.feature.melspectrogram(y=y, sr=sample_rate)
        decibels = librosa.amplitude_to_db(s, ref=np.max)
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(
            decibels, sr=sample_rate, hop_length=512, x_axis="time", y_axis="log"
        )
        plt.colorbar()
        plt.title("Mel spectrogram", fontsize=20)
        plt.show()
        return
