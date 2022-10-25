from typing import List

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


def show_training_and_validation_loss(
    epochs: range, loss_values: List[float], val_loss_values: List[float]
) -> None:
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def show_training_and_validation_accuracy(
    epochs: range, acc: List[float], val_acc: List[float]
) -> None:
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
