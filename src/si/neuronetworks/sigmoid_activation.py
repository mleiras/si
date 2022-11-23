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
        pass

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        return sigmoid_function(input_data)


