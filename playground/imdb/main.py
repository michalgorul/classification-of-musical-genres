from keras.callbacks import History

from playground.imdb.model.model import build_model
from playground.imdb.data.data import train_data, train_labels, test_data, test_labels
from playground.imdb.plot.plot import (
    show_training_and_validation_loss,
    show_training_and_validation_accuracy,
)
from playground.utils.utils import disable_cuda

# TODO: Try using one or three hidden layers, and see how doing so affects validation and test accuracy.
# TODO: Try using layers with more hidden units or fewer hidden units: 32 units, 64 units, and so on.
# TODO: Try using the mse loss function instead of binary_crossentropy.
# TODO: Try using the tanh activation (an activation that was popular in the early days of neural networks) instead of relu.

disable_cuda()

hidden_units = 16
iterations = 20
samples = 512

model = build_model(hidden_units)

history: History = model.fit(
    train_data,
    train_labels,
    epochs=iterations,
    batch_size=samples,
    validation_data=(test_data, test_labels),
)

loss_func_values = history.history.get("loss")
validation_loss_values = history.history.get("val_loss")
accuracy = history.history.get("val_binary_accuracy")
num_of_epochs = range(1, len(accuracy) + 1)

show_training_and_validation_loss(
    epochs=num_of_epochs, loss_values=loss_func_values, val_loss_values=validation_loss_values
)

show_training_and_validation_accuracy(
    epochs=num_of_epochs, acc=accuracy, val_acc=validation_loss_values
)

# Getting [val_binary_accuracy, accuracy]
results = model.evaluate(test_data, test_labels)
print(results)

# How model is confident in predictions
predictions = model.predictions(test_data)
print(predictions)
