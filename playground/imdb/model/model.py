from typing import Any

from keras import models, optimizers, metrics

from keras import layers
from keras import losses
from keras import activations


def build_model(hidden_units: int = 16) -> Any:
    """
    Function creating keras model from 1D vector. The intermediate layers will use relu as their
    activation function, and the final layer will use a sigmoid activation so as to output a
    probability (a score between 0 and 1, indicating how likely the sample is to have the target
    “1”: how likely the review is to be positive). A relu (rectified linear unit) is a function
    meant to zero out negative values, whereas a sigmoid “squashes” arbitrary values into the [0, 1]
    interval, outputting something that can be interpreted as a probability.
    :param hidden_units:
    :return:
    """
    model = models.Sequential()
    model.add(layers.Dense(hidden_units, activation=activations.relu, input_shape=(10000,)))
    model.add(layers.Dense(hidden_units, activation=activations.relu))
    model.add(layers.Dense(1, activation=activations.sigmoid))

    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=0.001),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy],
    )
    return model
