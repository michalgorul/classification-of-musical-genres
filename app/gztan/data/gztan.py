import datetime
import glob
import os
import random
import shutil
from typing import Tuple, Dict, List
from zipfile import ZipFile

import imageio
import librosa
import librosa.display
import numpy as np
import winsound
from matplotlib import pyplot as plt

from app.config import settings


class GtzanDataset:
    def __init__(self) -> None:
        self.gtzan_dataset_zip_path: str = settings.gtzan_zip_path
        self.gtzan_genres_original_path: str = settings.gtzan_genres_original
        self.gtzan_images_original_path: str = settings.gtzan_images_original

        self.directories: Dict[str, str] = {
            "train_dir": settings.gztan_train_dir,
            "val_dir": settings.gztan_validation_dir,
            "test_dir": settings.gztan_test_dir,
        }

        self.genres = [
            "blues",
            "classical",
            "country",
            "disco",
            "hiphop",
            "jazz",
            "metal",
            "pop",
            "reggae",
            "rock",
        ]

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

    def compare_created_to_read_spectogram(self, sound_file_path: str | None) -> None:
        wav_file_path, _ = self._get_original_file_path_and_name(file_path=sound_file_path)
        self.create_decibel_spectogram_from_sound_file(wav_file_path)
        self.create_mel_spectogram_from_sound_file(wav_file_path)

        image_file_path = (
            wav_file_path.replace("genres_original", "images_original")
            .replace(".", "")
            .replace("wav", ".png")
        )
        self.show_spectogram_from_dataset(image_file_path=image_file_path)
        return

    def _create_directories(self) -> None:
        # Create folders
        for folder_name, path in self.directories.items():
            if os.path.exists(path):
                shutil.rmtree(path)
                os.mkdir(path)
            else:
                os.mkdir(path)

    def _copy_files(self, file_paths: List[str], dest_dir: str) -> None:
        try:
            for file in file_paths:
                shutil.copy(
                    file,
                    os.path.join(
                        os.path.join(dest_dir),
                        os.path.split(file)[1],
                    ),
                )
        except KeyError as e:
            print(f"Failed to copy files to destination directory, error={e}")

    def data_init(self) -> None:
        self._create_directories()

        genres = list(os.listdir(self.gtzan_images_original_path))
        for genre in genres:
            # Finding all images & split in train, test, and validation
            src_file_paths = []

            for im in glob.glob(
                os.path.join(self.gtzan_images_original_path, f"{genre}", "*.png"), recursive=True
            ):
                src_file_paths.append(im)

            # Randomizing directories content
            random.shuffle(src_file_paths)

            test_files = src_file_paths[0:10]
            val_files = src_file_paths[10:20]
            train_files = src_file_paths[20:]

            #  make destination folders for train and test images
            for folder_name, path in self.directories.items():
                if not os.path.exists(path + f"\\{genre}"):
                    os.mkdir(f"{path}\\{genre}")

            # Coping training and testing images over
            self._copy_files(
                file_paths=train_files, dest_dir=f"{self.directories['train_dir']}\\{genre}\\"
            )
            self._copy_files(
                file_paths=test_files, dest_dir=f"{self.directories['test_dir']}\\{genre}\\"
            )
            self._copy_files(
                file_paths=val_files, dest_dir=f"{self.directories['val_dir']}\\{genre}\\"
            )

    def sanity_data_test(self) -> None:

        print("Genres directories in train data:", len(os.listdir(self.directories["train_dir"])))
        print("Genres directories in test data:", len(os.listdir(self.directories["test_dir"])))
        print(
            "Genres directories in validation data:", len(os.listdir(self.directories["val_dir"]))
        )

        print("\nTotal number of images in:")
        for genre in self.genres:
            for folder_name, path in self.directories.items():
                print(
                    f"\t{folder_name} of {genre} songs: " +
                    str(len(os.listdir(f"{self.directories[folder_name]}\\{genre}"))),
                )
            print()


gtzan: GtzanDataset = GtzanDataset()
