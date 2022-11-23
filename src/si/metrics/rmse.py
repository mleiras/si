import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    'test'
    pass


