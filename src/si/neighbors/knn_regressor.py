import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from data.dataset import Dataset
import numpy as np
from statistics.euclidean_distance import euclidean_distance
from metrics.rmse import rmse
from typing import Union


class KNNRegressor:
    def __init__(self, k: int, distance = euclidean_distance):
        self.k = k
        self.distance = distance
        
        self.dataset = None
    

    def fit(self, dataset: Dataset):
        self.dataset = dataset # dataset treino
        return self


    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:

        distances = self.distance(sample, self.dataset.X)
        
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors] # np.array com as varias classes

        return np.mean(k_nearest_neighbors_labels)


    def predict(self, dataset: Dataset):
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)


    def score(self, dataset: Dataset) -> float:
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)

    

if __name__ == '__main__':
    from model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN classifier
    knn = KNNRegressor(k=3)

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The accuracy of the model is: {score}')


    print('-----------------------------------')

    from io_folder.module_csv import read_csv
    
    cpu = read_csv('/home/monica/Documents/2_ano/sistemas/si/datasets/cpu.csv', sep=',',features=True, label=True)
    train_dataset, test_dataset = train_test_split(cpu)
    # print(test_dataset.X[:5])
    k = 2
    kmeans = KNNRegressor(k)
    kmeans.fit(train_dataset)
    predictions = kmeans.predict(test_dataset)
    print(predictions)
    print(kmeans.score(test_dataset))



    



    