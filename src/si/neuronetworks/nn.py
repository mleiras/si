import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from data.dataset import Dataset

class NN:
    def __init__(self, layers: list):
        self.layers = layers

    
    def fit(self, dataset = Dataset):
        x = dataset.X # temos que usar isto para depois poder substituir os valores de X (em cada iteração)
        for layer in self.layers:
            layer.forward(x)
        return self

    def predict(self, dataset: Dataset):
        pass




