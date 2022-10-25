from typing import Any, List, Dict, Sequence

import tensorflow
from pandas._typing import npt


def print_train_data(train_data: List[Any]) -> None:
    try:
        print(train_data[0])
        return
    except KeyError:
        print(f"Failed to print train data. The train_data[0] does not exist")


def print_test_data(test_data: List[Any]) -> None:
    try:
        print(test_data[0])
        return
    except KeyError:
        print(f"Failed to print test data. The test_data[0] does not exist")


def decode_imdb_reviews(train_data: List[Any], word_index: Dict[str, Any]) -> None:
    """
    :param train_data: train data consisting of list of integers (words)
    :param word_index: a dictionary mapping words to an integer index.
    :return:
    """
    try:
        if not word_index:
            raise TypeError
        # Reverses it, mapping integer indices to words
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        # Decodes the review. Note that the indices are offset by 3 because 0,
        # 1, and 2 are reserved indices for “padding,” “start of sequence,” and
        # “unknown.”
        decoded_review = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])
        print(decoded_review)
        return
    except KeyError:
        print(f"Failed to print decoded data. The train_data[0] does not exist")
    except TypeError:
        print(f"Failed to print decoded data. The word_index does not exist")


def disable_cuda() -> None:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
