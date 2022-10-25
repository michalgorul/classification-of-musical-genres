from typing import List, Any

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


# Replaced each point with an exponential moving average of the previous points, to obtain a smooth curve.
def smooth_curve(points: List[Any], factor: int = 0.9) -> List[Any]:
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def show_validation_mae(average_mae_history: List[Any]) -> None:
    smooth_mae_history = smooth_curve(average_mae_history[10:])
    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.ylabel("Validation MAE")
    plt.show()
    plt.xlabel("Epochs")
