from typing import Sequence, Any, List

import numpy as np
import numpy.typing as npt
from keras.datasets import reuters


# Reuters dataset, a set of short newswires and their
# topics, published by Reuters in 1986. It’s a simple, widely used toy dataset for
# text classification. There are 46 different topics; some topics are more
# represented than others, but each topic has at least 10 examples in the training
# set.


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


def to_one_hot(labels: npt.NDArray[Any], dimension: int = 46):
    """
    One-hot encoding is a widely used format for categorical data,
    also called categorical encoding. In this case, one-hot encoding
    of the labels consists of embedding each label as an all-zero vector
    with a 1 in the place of the label index. Also, to_categorical func from keras could be used
    :param labels: a tensor of labels
    :param dimension: number of dimensions, in this case number of topics in data
    :return:
    """
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.0
    return results


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

train_data = vectorize_sequences(train_data)
test_data = vectorize_sequences(test_data)
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)
