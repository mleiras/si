import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from data.dataset import Dataset

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
        mask = int(num*self.percentile/100)
        idx = np.argsort(self.F)[-mask:]
        best_features = dataset.X.iloc[:, idx] #tenho de usar iloc aqui
        best_features_names = [dataset.features[i] for i in idx]
        return Dataset(best_features, dataset.y, best_features_names, dataset.label)


    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

