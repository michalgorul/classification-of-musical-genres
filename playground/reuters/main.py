from keras.callbacks import History

from playground.reuters.data.data import (
    train_data,
    one_hot_train_labels,
    test_data,
    one_hot_test_labels,
)
from playground.reuters.model.model import build_model
from playground.reuters.plot.plot import (
    show_training_and_validation_loss,
    show_training_and_validation_accuracy,
)
from playground.utils.utils import disable_cuda

# TODO: Try another way to encode the labels (cast them as an integer tensor)
#   train_labels = np.array(train_labels)
#   test_labels = np.array(test_labels)
#   With integer labels, sparse_categorical_crossentropy should be used
# TODO: Try using larger or smaller layers: 32 units, 128 units, and so on.
# TODO: Try using g a single hidden layer, or three hidden layers.


disable_cuda()

hidden_units = 64
iterations = 10
samples = 512

model = build_model(hidden_units)

history: History = model.fit(
    train_data,
    one_hot_train_labels,
    epochs=iterations,
    batch_size=samples,
    validation_data=(test_data, one_hot_test_labels),
)

loss_func_values = history.history.get("loss")
validation_loss_values = history.history.get("val_loss")
accuracy = history.history.get("val_categorical_accuracy")
num_of_epochs = range(1, len(accuracy) + 1)

show_training_and_validation_loss(
    epochs=num_of_epochs, loss_values=loss_func_values, val_loss_values=validation_loss_values
)

show_training_and_validation_accuracy(
    epochs=num_of_epochs, acc=accuracy, val_acc=validation_loss_values
)

# Getting [val_loss, val_categorical_accuracy]
results = model.evaluate(test_data, one_hot_test_labels)
print(results)
