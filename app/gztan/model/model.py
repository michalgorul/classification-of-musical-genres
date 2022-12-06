from keras import models, layers, activations, optimizers, losses, metrics, Sequential
from keras.layers import Flatten


def build_model() -> Sequential:
    """
    Function creating keras model
    :return: a model
    """
    model = models.Sequential()

    model.add(layers.Conv2D(8, (3, 3), activation=activations.relu, input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(16, (3, 3), activation=activations.relu))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (3, 3), activation=activations.relu))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3, 3), activation=activations.relu))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), activation=activations.relu))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # flattening the data to be passed to a dense layer
    model.add(Flatten())

    model.add(layers.Dense(256, activation=activations.relu))
    model.add(layers.Dense(10, activation=activations.softmax))

    model.summary()

    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=1e-4),
        loss=losses.categorical_crossentropy,
        metrics=[metrics.categorical_accuracy],
    )

    return model
