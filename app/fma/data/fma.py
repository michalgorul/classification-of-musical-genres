import glob
import os
import random
import shutil
from typing import Dict, List, Any

import librosa
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from pandas import DataFrame

from app.config import settings
from app.fma import utils


class FmaDataset:
    def __init__(self) -> None:
        self.metadata_path = settings.fma_metadata_path
        self.genres_path = settings.fma_genres_path
        self.small_dataset = settings.fma_small_dataset_path
        self.images_path = settings.fma_images_path

        self.directories: Dict[str, str] = {
            "train_dir": settings.fma_train_dir,
            "val_dir": settings.fma_validation_dir,
            "test_dir": settings.fma_test_dir,
        }

    def load(self) -> DataFrame:
        tracks: DataFrame = utils.load(f"{self.metadata_path}/tracks.csv")
        return tracks

    def subset(self, tracks: DataFrame, subset_name: str) -> DataFrame:
        assert subset_name in ["small", "medium"]

        subset = tracks.index[tracks["set", "subset"] <= "small"]
        assert subset.isin(tracks.index).all()

        return tracks.loc[subset]

    def list_specific_genre_tracks(self, tracks: DataFrame, genre: str) -> None:
        tracks_with_genre_top = tracks.index[tracks["track", "genre_top"] == genre]
        print(genre.upper())
        print(tracks_with_genre_top.to_list())

    def get_track_ids_for_genre(self, tracks: DataFrame, genres: List[str]) -> Dict[str, List[Any]]:
        return {
            genre: tracks.index[tracks["track", "genre_top"] == genre].to_list() for genre in genres
        }

    def genres_top_track_ids(self, tracks: DataFrame) -> Dict[str, List[Any]]:
        subset_small = self.subset(tracks, "small")
        genres_top = list(tracks["track", "genre_top"].unique())
        return self.get_track_ids_for_genre(subset_small, genres_top)

    def _get_genres_top(self, tracks: DataFrame) -> List[str]:
        genres_top_fixed: List[str] = []
        try:
            genres_top = list(tracks["track", "genre_top"].unique())
            for genre in genres_top:
                if " " not in str(genre):
                    genres_top_fixed.append(genre)
                else:
                    genres_top_fixed.append(genre.split(" ")[0])
            print("Top genres got")
        except Exception as e:
            print(f"Failed to get top genres, error={e}")
        return genres_top_fixed

    def _create_directories(self, directories: List[str]) -> None:
        # Create folders
        for path in directories:
            if os.path.exists(path):
                shutil.rmtree(path)
                os.mkdir(path)
            else:
                os.mkdir(path)

    def _copy_files(self, file_paths: List[str], dest_dir: str) -> int:
        count = 0
        try:
            for file in file_paths:
                shutil.copy(
                    file,
                    os.path.join(
                        os.path.join(dest_dir),
                        os.path.split(file)[1],
                    ),
                )
                count += 1
        except KeyError as e:
            print(f"Failed to copy files to destination directory, error={e}")
        return count

    def _delete_dir_if_empty(self, dir_names: List[str]) -> None:
        dirs_to_remove = []
        count = 0
        for dir_name in dir_names:
            songs = glob.glob(
                os.path.join(self.genres_path, f"{dir_name}", "*.mp3"), recursive=True
            )
            if len(songs) == 0:
                dirs_to_remove.append(f"{self.genres_path}\\{dir_name}")

        print("Found empty directories:", len(dirs_to_remove))
        print("Removing found empty directories...")
        for path in dirs_to_remove:
            try:
                if os.path.exists(path):
                    shutil.rmtree(path)
                    count += 1
            except Exception as e:
                print(f"Failed to remove dir, dir={path}, error={e}")
        print("Removed empty directories:", count)

    def fill_directories_with_songs(self) -> None:
        tracks = self.load()
        ids = self.genres_top_track_ids(tracks)
        all_dirs = os.listdir(self.small_dataset)

        print("Getting Top genres...")
        genres_top = self._get_genres_top(tracks)

        print("Creating directories...")
        try:
            genres_dir_paths = [f"{self.genres_path}\\{genre}" for genre in genres_top]
            self._create_directories(genres_dir_paths)
            print("Directories created")
        except Exception as e:
            print(f"Failed to create directories, error={e}")

        print("Getting files to copy...")
        files_to_copy: Dict[str, List[str]] = {}
        try:
            files_to_copy = {f"{self.genres_path}\\{genre}": [] for genre in genres_top}
            for dir_name in all_dirs:
                for song_file in glob.glob(
                    os.path.join(self.small_dataset, f"{dir_name}", "*.mp3"), recursive=True
                ):
                    song_index = int(song_file.split("\\")[-1].split(".")[0])
                    for genre, ids_list in ids.items():
                        if " " in str(genre):
                            genre = genre.split(" ")[0]
                        if song_index in ids_list:
                            files_to_copy[f"{self.genres_path}\\{genre}"].append(song_file)
            print("Files to copy got")
        except Exception as e:
            print(f"Failed to get files to copy, error={e}")

        print("Copying files to desired directories...")
        try:
            files_copied = 0
            for genre_dir, songs in files_to_copy.items():
                print("\tCurrent genre:", genre_dir.split("\\")[-1])
                files_copied += self._copy_files(file_paths=songs, dest_dir=genre_dir)
            if files_copied == 0:
                raise ValueError("Zero files were copied")
            print("Files copied...")
        except Exception as e:
            print(f"Failed to get files to copy, error={e}")

        self._delete_dir_if_empty(os.listdir(self.genres_path))

    def make_spectograms(self, genre: str, last_song: int = 935) -> None:
        g = genre
        j = 0
        songs_path = self.genres_path
        images_path = self.images_path
        for filename in os.listdir(os.path.join(songs_path, f"{g}")):
            j = j + 1
            if j > last_song:
                print(f"Current file in {g}: {j}")

                song = os.path.join(f"{songs_path}\\{g}", f"{filename}")

                y, sr = librosa.load(song)
                # print(sr)
                mels = librosa.feature.melspectrogram(y=y, sr=sr)
                figure(figsize=(4, 2))
                plt.imshow(librosa.power_to_db(mels, ref=np.max), aspect="auto")
                plt.axis("off")

                genre_dir_path = f"{images_path}\\{g}"
                if not os.path.exists(genre_dir_path):
                    os.mkdir(genre_dir_path)

                plt.savefig(f"{genre_dir_path}\\{g + str(j)}.png")
                plt.close()

    def data_init(self) -> None:
        directories = self.directories
        self._create_directories(directories)

        images_path = self.images_path
        genres = list(os.listdir(images_path))
        for genre in genres:
            print(f"Current genre: {genre}")

            # Finding all images & split in train, test, and validation
            src_file_paths = []

            for file in glob.glob(os.path.join(images_path, f"{genre}", "*.png"), recursive=True):
                src_file_paths.append(file)

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


fma = FmaDataset()
