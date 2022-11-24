import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from model_selection.cross_validate import cross_validate
from data.dataset import Dataset


def randomized_search_cv(model, dataset: Dataset, parameter_grid: dict, scoring= None, cv: int = 5, n_iter: int = 10, test_size: float = 0.2) -> dict:

    #verificar se parametros existem com função hasattr
    for parameter in parameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} doesn't have this parameter")

    scores = []

    combinations = [] # lista de dicionarios (cada combinação de parametros)

    for i in range(n_iter):
        parameters = {}
        for parameter in parameter_grid:
            # print(parameter_grid[parameter])
            parameters[parameter] = np.random.choice(parameter_grid[parameter])
            # print(parameters[parameter])
        
        combinations.append(parameters)
            
    # obter combinações possiveis
    for combination in combinations:
        parameters = {}
        
        # # set do parametro
        for parameter, value in combination.items():
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
        'l2_penalty': np.linspace(1, 10, 10),
        'alpha': np.linspace(0.001, 0.0001, 100),
        'max_iter': np.linspace(1000, 2000, 200, dtype=int)
    }

    # cross validate the model
    scores = randomized_search_cv(modelo_lg,
                             breast_bin,
                             parameter_grid=parameter_grid_,
                             cv=3, n_iter=10)
    
    print(scores)

