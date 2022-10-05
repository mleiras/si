import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from data.dataset import Dataset


class VarianceThreshold:
    def __init__(self, threshold):
        if threshold < 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold # linha de corte/valor de corte
        self.variance = None

    def fit(self, dataset):
        self.variance = np.var(dataset.X, axis=0) ## ou então Dataset.get_var
        return self # retorna ele próprio


    def transform(self, dataset):
        mask = self.variance > self.threshold
        novo_X = dataset.X[:,mask]
        features= np.array(dataset.features)[mask] # selecionar as features daquelas que tem variance > threshold
        return Dataset(novo_X, y=dataset.y, features=list(features), label=dataset.label)


    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)



if __name__ == '__main__':
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    selector = VarianceThreshold(1)
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)