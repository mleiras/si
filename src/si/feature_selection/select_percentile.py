import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from data.dataset import Dataset
from statistics.f_classification import f_classification

class SelectPercentile():
    def __init__(self, score_func, percentile: int):
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def fit(self, dataset):
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset):
        num = len(dataset.features)
        mask = num*self.percentile
        idx = np.argsort(self.F)[-mask:]
        features= np.array(dataset.features)[idx] # selecionar as features com base nos idx 
        return Dataset(dataset.X[:,idx], y=dataset.y, features=list(features), label=dataset.label)


    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)



