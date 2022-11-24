import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np

class SoftMaxActivation:
    def __init__(self):
        pass

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        ezi = np.exp(input_data-input_data.max()) # ndarray.max() 
        return ezi / (np.sum(ezi, axis=1, keepdims=True))        
        


