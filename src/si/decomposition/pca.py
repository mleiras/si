import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from data.dataset import Dataset

class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components

        self.mean = None
        self.components = None
        self.explained_variance = None


    def __get_centered_data(self, dataset: Dataset):
        self.mean = np.mean(dataset.X, axis=0)
        centered_data = dataset.X - self.mean
        return centered_data


    def __get_svd(self, dataset: Dataset):
        U,S,Vt = np.linalg.svd(dataset, full_matrices=False)
        return U, S, Vt


    def fit(self, dataset: Dataset):
        # center the data
        centered_data = self.__get_centered_data(dataset)

        # get SVD
        U,S,Vt = self.__get_svd(centered_data)

        # principal components
        self.components = Vt[:,:self.n_components]

        # variance
        EV = (S **2) / (len(dataset.X)-1)
        self.explained_variance = EV[:self.n_components]

        return self


    def transform(self, dataset: Dataset):
        # center the data
        centered_data = self.__get_centered_data(dataset) # devia subtrair a media calculada anteriormente

        # X reduced
        *rest, Vt = self.__get_svd(centered_data)
        V = Vt.T # usar o V calculado anteriormente também
        X_red = np.dot(centered_data, V)

        return X_red


    def fit_transform(self, dataset: Dataset):
        self.fit(dataset)
        return self.transform(dataset)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from io_folder.module_csv import read_csv 
    iris = read_csv('/home/monica/Documents/2_ano/sistemas/si/datasets/iris.csv', sep=',',features=True, label=True)
    print(iris.X[:5])
    n = 2
    iris_pca = PCA(n)
    iris_x_red = iris_pca.fit_transform(iris)
    print(iris_pca.explained_variance)

    plt.bar(range(n), iris_pca.explained_variance*100)
    plt.xticks(range(n), ['PC'+str(i) for i in range(1,n+1)])
    plt.title("Variância explicada por PC")
    plt.ylabel("Percentagem")
    plt.show()




