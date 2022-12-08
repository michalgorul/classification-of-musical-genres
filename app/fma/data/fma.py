from typing import Dict, Tuple

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

from app.config import settings
from app.fma import utils


class FmaDataset:
    def __init__(self) -> None:
        self.metadata_path = settings.fma_metadata_path

        self.directories: Dict[str, str] = {
            "train_dir": settings.fma_train_dir,
            "val_dir": settings.fma_validation_dir,
            "test_dir": settings.fma_test_dir,
        }

    def load(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        tracks: DataFrame = utils.load(f'{self.metadata_path}/tracks.csv')
        features: DataFrame = utils.load(f'{self.metadata_path}/features.csv')
        echonest: DataFrame = utils.load(f'{self.metadata_path}/echonest.csv')

        np.testing.assert_array_equal(features.index, tracks.index)
        assert echonest.index.isin(tracks.index).all()

        print("tracks:", tracks.shape, "features:", features.shape, "echonest:", echonest.shape)

        return tracks, features, echonest

    def subset(self) -> None:
        tracks, features, echonest = self.load()
        subset = tracks.index[tracks['set', 'subset'] <= 'small']

        assert subset.isin(tracks.index).all()
        assert subset.isin(features.index).all()

        features_all = features.join(echonest, how='inner').sort_index(axis=1)
        print('Not enough Echonest features: {}'.format(features_all.shape))

        tracks = tracks.loc[subset]
        features_all = features.loc[subset]

        print("tracks small shape:", tracks.shape, "features small shape:", features_all.shape)

        train = tracks.index[tracks['set', 'split'] == 'training']
        val = tracks.index[tracks['set', 'split'] == 'validation']
        test = tracks.index[tracks['set', 'split'] == 'test']

        print(test)

        print('{} training examples, {} validation examples, {} testing examples'.format(*map(len, [train, val, test])))

        genres = list(LabelEncoder().fit(tracks['track', 'genre_top']).classes_)
        # genres = list(tracks['track', 'genre_top'].unique())
        print('Top genres ({}): {}'.format(len(genres), genres))
        genres = list(MultiLabelBinarizer().fit(tracks['track', 'genres_all']).classes_)
        print('All genres ({}): {}'.format(len(genres), genres))


fma = FmaDataset()

