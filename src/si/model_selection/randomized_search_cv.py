import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from io_folder.module_csv import read_csv

import numpy as np
from cross_validate import cross_validate
from data.dataset import Dataset
import itertools


def randomized_search_cv(model, dataset: Dataset, parameter_distribution: dict, scoring= None, cv: int = 5, n_iter: int = 10, test_size: float = 0.2) -> dict:

    #verificar se parametros existem com função hasattr
    for parameter in parameter_distribution:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} doesn't have this parameter")

    scores = []

    # n_iter ??

    # corrigir isto
    
    # obter combinações possiveis
    for combination in itertools.product(*parameter_grid.values()):
        parameters = {}

        # set do parametro
        for parameter, value in zip(parameter_grid.keys(), combination):
            setattr(model, parameter, value)
            parameters[parameter] = value

        # cross validation do modelo com os parametros da combinação
        score = cross_validate(model = model, dataset = dataset, scoring=scoring, cv = cv, test_size = test_size)

        #adicionar aos scores a lista de parametros da combinação
        scores['parameters'].append(parameters)

        scores.append(score)

    return scores


if __name__ == '__main__':

    breast_bin = read_csv('/home/monica/Documents/2_ano/sistemas/si/datasets/breast_bin.csv')


