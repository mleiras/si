import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from metrics.accuracy import accuracy
from io_folder.module_csv import read_csv

class VotingClassifier:
    def __init__(self, models):
        self.models = models # lista de modelos


    def fit(self, dataset):
        for model in self.models:
            model.fit(dataset)
        return self


    def predict(self, dataset):
        # lista = []
        # for model in self.models:
        #     lista.append(model.predict(dataset))
        
        def _get_majority_vote(pred):

            labels, counts = np.unique(pred, return_counts=True)
            return labels[np.argmax(counts)]
        
        predictions = np.array([model.predict(dataset) for model in self.models]).transpose() # para usar o axis = 1 temos de transpor
        return np.apply_along_axis(_get_majority_vote, axis = 1, arr=predictions)

    def score(self, dataset):
        return accuracy(dataset.y, self.predict(dataset))


    
if __name__ == '__main__':
    breast_bin = read_csv('/home/monica/Documents/2_ano/sistemas/si/datasets/breast-bin.csv', sep=',', features = False, label=True)
    


