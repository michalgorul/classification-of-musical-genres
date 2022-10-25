import numpy as np

from playground.boston.data.data import train_data
from playground.boston.data.data import train_targets, test_data, test_targets
from playground.boston.model.model import build_model
from playground.boston.plot.plot import show_validation_mae
from playground.utils.utils import disable_cuda

disable_cuda()

k = 4
num_validation_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []

for i in range(k):
    print("processing fold #", i)
    # Prepares the validation data: data from partition #k
    val_data = train_data[i * num_validation_samples : (i + 1) * num_validation_samples]
    val_targets = train_targets[i * num_validation_samples : (i + 1) * num_validation_samples]

    # Prepares the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[: i * num_validation_samples], train_data[(i + 1) * num_validation_samples :]],
        axis=0,
    )
    partial_train_targets = np.concatenate(
        [
            train_targets[: i * num_validation_samples],
            train_targets[(i + 1) * num_validation_samples :],
        ],
        axis=0,
    )
    # Builds the Keras model (already compiled)
    model = build_model()

    # Trains the model (in silent mode, verbose=0)
    history = model.fit(
        partial_train_data,
        partial_train_targets,
        validation_data=(val_data, val_targets),
        epochs=num_epochs,
        batch_size=1,
    )
    mae_history = history.history["val_mean_absolute_error"]
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

show_validation_mae(average_mae_history)


# Gets a fresh, compiled model
model = build_model()

# Trains it on the entirety of the data
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

# Final result
print(test_mae_score)
