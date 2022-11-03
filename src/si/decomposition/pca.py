import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from data.dataset import Dataset

class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components

        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset):
        
        return self


    def transform(self):
        pass