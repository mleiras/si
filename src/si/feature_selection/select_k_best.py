import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from data.dataset import Dataset
from statistics.f_classification import f_classification


class SelectKBest:
    def __init__(self, score_func, k: int):
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def fit(self, dataset):
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset):
        idx = np.argsort(self.F)[-self.k:]
        features= np.array(dataset.features)[idx] # selecionar as features com base nos idx 
        return Dataset(dataset.X[:,idx], y=dataset.y, features=list(features), label=dataset.label)


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

    selector = SelectKBest(f_classification, 1)
    dataset = selector.fit_transform(dataset)
    print(dataset.features)