from re import A
import sys
from tkinter.filedialog import asksaveasfile
sys.path.insert(0, '/home/monica/Documents/2_ano/sistemas/si/src/si')
import numpy as np
from data.dataset import Dataset


def read_data_file(filename: str, sep: str, label: bool or list):
    data = np.genfromtxt(filename, delimiter=sep, names=label)
    if label is True:
        pass
    y = None
    features = None
    return Dataset(data, y, features, label)


def write_data_file():
    pass


if __name__ == '__main__':
    teste = read_data_file('si/datasets/breast-bin.data', sep=',', label=None)
    # teste = read_data_file('si/datasets/iris.csv', sep=',', label=True)
    print(teste.shape())
    print(teste.y)
    print(teste.X)
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
