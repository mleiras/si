import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from cross_validate import cross_validate
from data.dataset import Dataset
import itertools


def grid_search_cv(model, dataset: Dataset, parameter_grid: dict, scoring= None, cv: int = 5, test_size: float = 0.2) -> dict:

    #verificar se parametros existem com função hasattr
    for parameter in parameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} doesn't have this parameter")

    scores = []

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
        score['parameters'] = parameters

        scores.append(score)
    
    return scores


if __name__ == '__main__':
    from io_folder.module_csv import read_csv
    from sklearn.preprocessing import StandardScaler
    from linear_model.logistic_regression import LogisticRegression

    breast_bin = read_csv('/home/monica/Documents/2_ano/sistemas/si/datasets/breast-bin.csv', sep=',', features = False, label=True)
    breast_bin.X = StandardScaler().fit_transform(breast_bin.X)
    modelo_lg = LogisticRegression()
    
    parameter_grid_ = {
        'l2_penalty': (1, 10),
        'alpha': (0.001, 0.0001),
        'max_iter': (1000, 2000)
    }

    # cross validate the model
    scores = grid_search_cv(modelo_lg,
                             breast_bin,
                             parameter_grid=parameter_grid_,
                             cv=3)
    
    print(scores)
