import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from metrics.accuracy import accuracy
from io_folder.module_csv import read_csv

class StackingClassifier:
    def __init__(self, models, final_model):
        self.models = models # lista de modelos
        self.final_model = final_model


    def fit(self, dataset):
        # fit para cada modelo da lista
        for model in self.models:
            model.fit(dataset)
        # guardar as previsões de cada modelo
        predictions = np.array([model.predict(dataset) for model in self.models]) # é preciso fazer transposta aqui? ver depois nos testes
        # fit do modelo final com as previsões calculadas anteriormente
        self.final_model.fit(predictions)        
        return self


    def predict(self, dataset):
        # guardar as previsões de cada modelo
        predictions = np.array([model.predict(dataset) for model in self.models])  # é preciso fazer transposta aqui? ver depois nos testes
        # fit do modelo final com as previsões calculadas anteriormente
        return self.final_model.predict(predictions)

    def score(self, dataset):
        return accuracy(dataset.y, self.predict(dataset))

    
if __name__ == '__main__':
    breast_bin = read_csv('/home/monica/Documents/2_ano/sistemas/si/datasets/breast-bin.csv', sep=',', features = False, label=True)
    


