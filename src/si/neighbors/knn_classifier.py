import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from data.dataset import Dataset
import pandas as pd
import numpy as np
from statistics.euclidean_distance import euclidean_distance
from metrics.accuracy import accuracy

class KNNClassifier:

    def __init__(self, k: int, distance = euclidean_distance):
            self.k = k
            self.distance = distance
            self.dataset = None
    

    def fit(self, dataset: Dataset):
        self.dataset = dataset # dataset treino
        return self

    
    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:

        distances = self.distance(sample, self.dataset)
        
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors] # np.array com as varias classes

        labels, counts = np.unique(k_nearest_neighbors, k_nearest_neighbors_labels, return_counts=True)
    
        return labels[np.argmax(counts)]


    def predict(self, dataset: Dataset):
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)


    def score(self, dataset: Dataset) -> float:
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)
