from typing import Any

from keras import activations
from keras import layers
from keras import losses
from keras import models, optimizers, metrics

from playground.boston.data.data import train_data


def build_model(hidden_units: int = 64) -> Any:
    """
    Function creating keras model from 1D vector.
    :param hidden_units: number of hidden layers
    :return: a model
    """
    model = models.Sequential()
    model.add(layers.Dense(hidden_units, activation=activations.relu, input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(hidden_units, activation=activations.relu))
    # No activation (linear layer) -> scalar regression
    model.add(layers.Dense(1))

    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=0.001),
        # mse -> Mean squared error (the square of the difference between the predictions and the targets)
        loss=losses.mse,
        # mae -> Mean absolute error (absolute value of the difference between the predictions andthe targets)
        # For instance, an MAE of 0.5 on this problem would mean your predictions are off by $500 on average.)
        metrics=[metrics.mae],
    )
    return model
