import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 



from data.dataset import Dataset
import pandas as pd
import numpy as np


def read_csv(filename: str, sep: str, features: bool, label: bool):

    data = pd.read_csv(filename, sep)
    if features and label:  # se tiver nomes nas colunas e a variavel y
        y = data.iloc[:, -1] #.to_numpy() # para passar para array 
        label = data.columns[-1]
        data = data.iloc[:, :-1]
        features = data.columns
        dataset = Dataset(data, y=y, features=features, label=label)

    elif features and label is False:  # se tiver nomes nas colunas mas não tem y
        features = data.columns
        dataset = Dataset(data, features=features)

    elif features is False and label:  # se tiver y mas não tem nomes nas colunas
        y = data.iloc[:, -1]
        label = data.columns[-1]
        data = data.iloc[:, :-1]
        dataset = Dataset(data, y=y, label=label)

    else:  # quando não tem nem nomes nas colunas nem tem y nos dados
        dataset = Dataset(data)

    return dataset


def write_csv(filename, dataset, sep, features, label):
    data = pd.DataFrame(dataset.X)

    if features:
        data.columns = dataset.features


    if label:
        data[dataset.label] = dataset.y

    data.to_csv(filename, sep=sep, index = False)



if __name__ == '__main__':
    teste = read_csv('si/datasets/iris.csv', sep=',',
                     features=True, label=True)
    print(teste.X.head(10))
    print('---------------------------------------------------------------')
    print('Shape: \n', teste.shape())
    print('Label: \n', teste.has_label())
    print('Classes: \n', teste.get_classes())
    print('Média: \n', teste.get_mean())
    print('Variância: \n', teste.get_variance())
    print('Mediana: \n', teste.get_median())
    print('Min: \n', teste.get_min())
    print('Max: \n', teste.get_max())
    print('Summary: \n', teste.summary())
