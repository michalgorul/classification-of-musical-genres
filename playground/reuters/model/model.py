from typing import Any

from keras import layers
from keras import models, activations, optimizers, losses
from keras.metrics import metrics


def build_model(hidden_units: int = 64) -> Any:
    """
    Function creating keras model from 1D vector. The intermediate layers will use relu as their
    activation function. Ending with Dense layer of size 46. This means for each input sample,
    the network will output a 46-dimensional vector. Each entry in this vector (each dimension)
    will encode a different output class. The last layer uses a softmax activation.  It means the network
    will output a probability distribution over the 46 different output classes—for every input sample,
    the network will produce a 46-dimensional output vector, where output[i] is the probability
    that the sample belongs to class i. The 46 scores will sum to 1.
    :param hidden_units: number of hidden layers
    :return: a model
    """
    model = models.Sequential()
    model.add(layers.Dense(hidden_units, activation=activations.relu, input_shape=(10000,)))
    model.add(layers.Dense(hidden_units, activation=activations.relu))
    model.add(layers.Dense(46, activation=activations.softmax))

    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=0.001),
        loss=losses.categorical_crossentropy,
        metrics=[metrics.categorical_accuracy],
    )
    return model
