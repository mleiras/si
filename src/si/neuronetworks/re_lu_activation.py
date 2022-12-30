import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np

class ReLUActivation:
    def __init__(self):
        self.input_data = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input_data = input_data
        return np.maximum(0, input_data)

    # def __sub_val__(self, sample: np.ndarray) -> np.ndarray:
    #     if sample < 0:
    #         sample = 0
    #     else:
    #         sample = 1
    #     return sample

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        # substituir valores de self.input_data (inferiores a 0 por 0 e superiores a 0 por 1) # pode-se utilizar função where
        self.input_data = np.where(self.input_data <0, 0,1)

        error_to_propagate = error * self.input_data

        return error_to_propagate