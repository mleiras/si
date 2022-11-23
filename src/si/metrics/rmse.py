import numpy as np
import math

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(np.sum((y_true - y_pred)**2)/len(y_true))

