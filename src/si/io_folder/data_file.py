import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from data.dataset import Dataset
import pandas as pd
import numpy as np


def read_data_file(filename: str, sep: str, label: bool):
    data = np.genfromtxt(filename, delimiter=sep)
    if label:
        X = data[:, :-1]
        y = data[:, -1]
    else:
        X = data
        y = None

    return Dataset(data, y)


def write_data_file():
    pass


if __name__ == '__main__':
    teste = read_data_file('si/datasets/breast-bin.data', sep=',', label=False)
    # teste = read_data_file('si/datasets/iris.csv', sep=',', label=True)
    print(teste.shape())
    print(teste.y)
    print(teste.X)
    print('---------------------------------------------------------------')
    print('Shape: \n', teste.shape())
    print('Label: \n', teste.has_label())
    # print('Classes: \n', teste.get_classes()) # apenas com label=True
    print('Média: \n', teste.get_mean())
    print('Variância: \n', teste.get_variance())
    print('Mediana: \n', teste.get_median())
    print('Min: \n', teste.get_min())
    print('Max: \n', teste.get_max())
    print('Summary: \n', teste.summary())
