import numpy as np

def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ...
    
    return -2*(y_pred * (y_true)) / len(y_true) # corrigir
