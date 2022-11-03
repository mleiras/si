import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from data.dataset import Dataset
import pandas as pd
import numpy as np

def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> tuple:
    np.random.seed(random_state)

    n_samples = dataset.shape()[0]
    n_test = int(n_samples * test_size)

    permutations = np.random.permutation(n_samples)

    test_idx = permutations[:n_test]

    train_idx = permutations[n_test:]

    train = Dataset(dataset.X[train_idx], dataset.y[train_idx], features=dataset.features, label=dataset.label)

    test = Dataset(dataset.X[test_idx], dataset.y[test_idx], features=dataset.features, label=dataset.label)

    return train, test
