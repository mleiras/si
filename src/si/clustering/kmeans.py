import os
import sys
import inspect
from typing import Callable

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from data.dataset import Dataset
from statistics.euclidean_distance import euclidean_distance

class KMeans:
    def __init__(self, k: int, max_iter: int = 1000, distance: Callable = euclidean_distance):
        self.k = k
        self.max_iter = max_iter
        self.distance = distance

        self.centroids = None
        self.labels = None

    def __centroids__(self, dataset):
        seeds = np.random.permutation(dataset.shape()[0])[:self.k]
        self.centroids = dataset.X[seeds]


    def __get_closest_centroid__(self, sample: np.ndarray) -> np.ndarray:
        centroids_distances = self.distance(sample, self.centroids)
        closest_centroids_idx = np.argmin(centroids_distances, axis=0) #retorna o indice que tem o menor valor no vetor das distancias aos centroides (ou seja, o centroide mais proximo para aquela amostra)
        return closest_centroids_idx


    def fit(self, dataset):

        convergence = False
        i = 0
        labels = np.zeros() # corrigir aqui
        while not convergence and i< self.max_iter:
            centroids = []
            new_labels = np.apply_along_axis(self.__get_closest_centroid__, axis=1, arr=dataset.X)
            for i in range(self.k):
                centroid = dataset.X[labels == i]
                centroid_mean = np.mean(centroid)
                centroids.append(centroid_mean)
            self.centroids = np.array(centroids)

            convergence = np.any(new_labels != labels)

            labels = new_labels

            i += 1

        self.labels = labels
        return self


    def __get_distances__(self, sample: np.ndarray) -> np.ndarray:
        return self.distance(sample, self.centroids) # esta função adicional é util porque o np.apply só permite colocar um array enquanto aqui podemos colocar os 2

    def trasnform(self, dataset: Dataset) -> np.ndarray:
        centroids_distances = np.apply_along_axis(self.__get_distances__, axis=1, arr=dataset.X)
        return centroids_distances


    def predict(self, dataset: Dataset) -> np.ndarray:
        return np.apply_along_axis(self.__get_closest_centroid__, axis=1, arr=dataset.X)







## nota para SVD: X,S,Vt = np.linalg(X)