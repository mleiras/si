import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from statistics.sigmoid_function import sigmoid_function

class SigmoidActivation:
    def __init__(self):
        self.input_data = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input_data = input_data
        return sigmoid_function(input_data)


    def backward(self, error: np.ndarray) -> np.ndarray:
        deriv_sig = sigmoid_function(self.input_data) * (1-sigmoid_function(self.input_data))

        error_to_propagate = error * deriv_sig
        
        return error_to_propagate