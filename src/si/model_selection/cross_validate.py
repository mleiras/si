import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from io_folder.module_csv import read_csv

import numpy as np
from split import train_test_split
from data.dataset import Dataset


def cross_validate(model, dataset: Dataset, scoring= None, cv: int = 3, test_size: float = 0.2) -> dict:

    scores = {
        'seeds': [],
        'train': [],
        'test': []
    }

    for i in cv:
        seed = np.random.randint(0, 1000)
        scores['seeds'].append(seed)
        data_train, data_test = train_test_split(dataset, test_size, random_state=seed)

        model.fit(data_train)

        if scoring is None:
            scores['train'].append(model.score(data_train))

            scores['test'].append(model.score(data_test))

        else:
            y_train = data_train.y
            y_test = data_test.y

            scores['train'].append(scoring(y_train, model.predict(y_train)))

            scores['test'].append(scoring(y_test, model.predict(y_test)))

    
    return scores



if __name__ == '__main__':

    breast_bin = read_csv('/home/monica/Documents/2_ano/sistemas/si/datasets/breast_bin.csv')


