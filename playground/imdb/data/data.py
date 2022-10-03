from typing import Sequence, List, Any

import numpy as np
import numpy.typing as npt
from keras.datasets import imdb


def vectorize_sequences(sequences: Sequence[List[int]], dimension: int = 10000) -> npt.NDArray[Any]:
    """
    Function turning lists into tensors
    :param sequences: integer sequences
    :param dimension: number of dimensions of result vector (1D tensors)
    :return: tensor made up of sequence
    """
    # Creates an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    # Sets specific indices of results[i] to 1s
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def vectorize_simple_list(labels: List[int]) -> npt.NDArray[Any]:
    """
    Function turning labels into tensor
    :param labels: list of int labels
    :return: tensor of int labels
    """
    return np.asarray(labels).astype("float32")


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

train_data = vectorize_sequences(train_data)
test_data = vectorize_sequences(test_data)
train_labels = vectorize_simple_list(train_labels)
test_labels = vectorize_simple_list(test_labels)
