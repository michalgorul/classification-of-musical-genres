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
from keras.callbacks import History
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
from pydub import AudioSegment

from app.config import settings
from app.gztan.data.data import get_train_data_generator, get_validation_data_generator
from app.gztan.model.model import build_model
from app.plotting.plot import (
    show_training_and_validation_loss,
    show_training_and_validation_accuracy,
)


class GtzanDataset:
    def __init__(self) -> None:
        self.gtzan_dataset_zip_path: str = settings.gtzan_zip_path
        self.gtzan_genres_original_path: str = settings.gtzan_genres_original
        self.gtzan_genres_3_sec_original: str = settings.gtzan_genres_3_sec_original
        self.gtzan_images_original_path: str = settings.gtzan_images_original
        self.gtzan_images_3_sec_original: str = settings.gtzan_images_3_sec_original

        self.directories_10sec: Dict[str, str] = {
            "train_dir": settings.gztan_train_dir,
            "val_dir": settings.gztan_validation_dir,
            "test_dir": settings.gztan_test_dir,
        }

        self.directories_3sec: Dict[str, str] = {
            "train_dir": settings.gztan_train_3sec_dir,
            "val_dir": settings.gztan_validation_3sec_dir,
            "test_dir": settings.gztan_test_3sec_dir,
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

    def _create_directories(self, directories: Dict[str, str]) -> None:
        # Create folders
        for folder_name, path in directories.items():
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

    def make_3_sec_wavs(self) -> None:
        genres = list(os.listdir(self.gtzan_genres_original_path))
        i = 0
        for genre in genres:
            print(f"Current genre: {genre}")
            # Finding all wavs & create 3sec songs
            src_file_paths = []

            path = self.gtzan_genres_original_path
            for im in glob.glob(os.path.join(path, f"{genre}", "*.wav"), recursive=True):
                src_file_paths.append(im)
            j = 0
            for song in src_file_paths:
                # print(f"Current song: {song}")

                j = j + 1
                for w in range(0, 10):
                    try:
                        i = i + 1
                        t1 = 3 * w * 1000
                        t2 = 3 * (w + 1) * 1000
                        new_audio = AudioSegment.from_wav(song)
                        new = new_audio[t1:t2]

                        genre_dir_path = f"{self.gtzan_genres_3_sec_original}\\{genre}"
                        if not os.path.exists(genre_dir_path):
                            os.mkdir(genre_dir_path)

                        file_name = genre + str(j) + str(w)
                        new.export(
                            f"{genre_dir_path}\\{file_name}.wav",
                            format="wav",
                        )
                    except Exception:
                        pass

    def make_3_sec_images(self, genre: str) -> None:
        # genres = list(os.listdir(self.gtzan_genres_3_sec_original))
        #
        # for g in genres:
        g = genre
        j = 0
        for filename in os.listdir(os.path.join(self.gtzan_genres_3_sec_original, f"{g}")):
            j = j + 1
            print(f"Current file in {g}: {j}")

            song = os.path.join(f"{self.gtzan_genres_3_sec_original}\\{g}", f"{filename}")

            y, sr = librosa.load(song, duration=3)
            # print(sr)
            mels = librosa.feature.melspectrogram(y=y, sr=sr)
            fig = plt.Figure()
            canvas = FigureCanvas(fig)
            p = plt.imshow(librosa.power_to_db(mels, ref=np.max))

            genre_dir_path = f"{self.gtzan_images_3_sec_original}\\{g}"
            if not os.path.exists(genre_dir_path):
                os.mkdir(genre_dir_path)
            plt.savefig(f"{genre_dir_path}\\{g + str(j)}.png")

    def data_init(self, sec_3: bool = False) -> None:
        directories = self.directories_3sec if sec_3 else self.directories_10sec
        self._create_directories(directories)

        images_path = self.gtzan_images_3_sec_original if sec_3 else self.gtzan_images_original_path
        genres = list(os.listdir(images_path))
        for genre in genres:
            print(f"Current genre: {genre}")

            # Finding all images & split in train, test, and validation
            src_file_paths = []

            for im in glob.glob(os.path.join(images_path, f"{genre}", "*.png"), recursive=True):
                src_file_paths.append(im)

            # Randomizing directories content
            random.shuffle(src_file_paths)

            test_files = src_file_paths[0:50]
            val_files = src_file_paths[50:200]
            train_files = src_file_paths[20:]

            #  make destination folders for train and test images
            for folder_name, path in directories.items():
                if not os.path.exists(path + f"\\{genre}"):
                    os.mkdir(f"{path}\\{genre}")

            # Coping training and testing images over
            self._copy_files(
                file_paths=train_files, dest_dir=f"{directories['train_dir']}\\{genre}\\"
            )
            self._copy_files(
                file_paths=test_files, dest_dir=f"{directories['test_dir']}\\{genre}\\"
            )
            self._copy_files(file_paths=val_files, dest_dir=f"{directories['val_dir']}\\{genre}\\")

    def sanity_data_test(self) -> None:

        print(
            "Genres directories in train data:",
            len(os.listdir(self.directories_10sec["train_dir"])),
        )
        print(
            "Genres directories in test data:", len(os.listdir(self.directories_10sec["test_dir"]))
        )
        print(
            "Genres directories in validation data:",
            len(os.listdir(self.directories_10sec["val_dir"])),
        )

        print("\nTotal number of images in:")
        for genre in self.genres:
            for folder_name, path in self.directories_10sec.items():
                print(
                    f"\t{folder_name} of {genre} songs: "
                    + str(len(os.listdir(f"{self.directories_10sec[folder_name]}\\{genre}"))),
                )
            print()

    def train_gtzan(self) -> None:
        # gtzan.make_3_sec_wavs()
        # gtzan.data_init(sec_3=True)

        # gtzan.data_init()
        # gtzan.sanity_data_test()

        train_data = get_train_data_generator(False)
        val_data = get_validation_data_generator(False)

        model = build_model(False)

        history: History = model.fit(
            train_data,
            steps_per_epoch=train_data.samples / train_data.batch_size,
            epochs=80,
            validation_data=val_data,
            validation_steps=val_data.samples / val_data.batch_size,
        )

        model.save("gztan/model/gztan.h5")
        #
        train_loss_values = history.history.get("loss")
        val_loss_values = history.history.get("val_loss")
        train_accuracy = history.history.get("categorical_accuracy")
        val_accuracy = history.history.get("val_categorical_accuracy")
        num_of_epochs = range(1, len(train_accuracy) + 1)

        print(f"categorical_accuracy max: {max(history.history.get('categorical_accuracy'))}")
        print(
            f"val_categorical_accuracy max: {max(history.history.get('val_categorical_accuracy'))}"
        )
        print(f"loss min: {min(history.history.get('loss'))}")
        print(f"val_loss min: {min(history.history.get('val_loss'))}")

        show_training_and_validation_loss(
            epochs=num_of_epochs, loss_values=train_loss_values, val_loss_values=val_loss_values
        )

        show_training_and_validation_accuracy(
            epochs=num_of_epochs, acc=train_accuracy, val_acc=val_accuracy
        )


gtzan: GtzanDataset = GtzanDataset()
