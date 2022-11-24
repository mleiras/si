import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from data.dataset import Dataset

class NN:
    def __init__(self, layers: list):
        self.layers = layers

    
    def fit(self, dataset = Dataset):
        x = dataset.X # temos que usar isto para depois poder substituir os valores de X (em cada iteração)
        for layer in self.layers:
            x = layer.forward(x)
        
        return self

    
    def predict(self, dataset: Dataset):
        pass



if __name__ == '__main__':
    from neuronetworks.layer import Dense
    from neuronetworks.sigmoid_activation import SigmoidActivation
    from neuronetworks.soft_max_activation import SoftMaxActivation
    from neuronetworks.re_lu_activation import ReLUActivation

    X = np.array([[0,0],
                [0,1],
                [1,0],
                [1,2]])

    Y = np.array([1,
                0,
                0,
                1])

    dataset = Dataset(X, Y, ['x1', 'x2'], 'x1 XNOR x2')

    l1 = Dense(input_size = 2, output_size=2)
    l2 = Dense(input_size = 2, output_size=1)

    l1_sa = SigmoidActivation()
    l2_sa = SigmoidActivation()

    nn_model_sa = NN(layers=[ l1, l1_sa, l2, l2_sa])
    nn_model_sa.fit(dataset=dataset)

    l1_sma = SoftMaxActivation()
    l2_sma = SoftMaxActivation()

    nn_model_sma = NN(layers=[ l1, l1_sma, l2, l2_sma])
    nn_model_sma.fit(dataset=dataset)

    l1_rlua = ReLUActivation()
    l2_rlua = ReLUActivation()

    nn_model_rlua = NN(layers=[ l1, l1_rlua, l2, l2_rlua])
    nn_model_rlua.fit(dataset=dataset)



