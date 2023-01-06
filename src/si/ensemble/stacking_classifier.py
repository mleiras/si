import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from data.dataset import Dataset
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
        model_pred = Dataset(predictions.T, dataset.y)
        self.final_model.fit(model_pred)
        return self


    def predict(self, dataset):
        # guardar as previsões de cada modelo
        predictions = np.array([model.predict(dataset) for model in self.models])  # é preciso fazer transposta aqui? ver depois nos testes
        # fit do modelo final com as previsões calculadas anteriormente
        model_pred = Dataset(dataset.X, predictions.T)

        return self.final_model.predict(model_pred)


    def score(self, dataset):
        return accuracy(dataset.y, self.predict(dataset))

    
if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler
    from model_selection.split import train_test_split
    from neighbors.knn_classifier import KNNClassifier
    from linear_model.logistic_regression import LogisticRegression

    breast_bin = read_csv('/home/monica/Documents/2_ano/sistemas/si/datasets/breast-bin.csv', sep=',', features = False, label=True)
    breast_bin.X = StandardScaler().fit_transform(breast_bin.X)
    breast_bin_train, breast_bin_test = train_test_split(breast_bin)
    modelo_knn = KNNClassifier()
    modelo_lg = LogisticRegression()
    modelo_final = KNNClassifier(3)
    stacking_model = StackingClassifier([modelo_knn, modelo_lg], modelo_final)
    stacking_model.fit(breast_bin_train)
    print(stacking_model.score(breast_bin_test))

