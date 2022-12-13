import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from data.dataset import Dataset
from metrics.mse import mse
from metrics.mse_derivative import mse_derivative
from metrics.accuracy import accuracy

from typing import Callable


class NN:
    def __init__(self, layers: list, epochs: int = 1000, loss_function: Callable = mse, learning_rate: float = 0.01, loss_derivative: Callable = mse_derivative, verbose: bool = False):
        self.layers = layers
        self.epochs = epochs # nº iterações
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.loss_derivative = loss_derivative
        self.verbose = verbose

        self.history = {}

    error
    def fit(self, dataset = Dataset) -> 'NN':
        

        for epoch in range(self.epochs):
            y_pred = np.array(dataset.X)
            y_true = np.reshape(dataset.y, (-1,1))

            for layer in self.layers:
                y_pred = layer.forward(y_pred)
            
            error = self.loss_derivative(y_true, y_pred)

            for layer in self.layers[::-1]: # começamos pela ultima layer
                error = layer.backward(error, self.learning_rate)
            
            # para saber se estamos a chegar ao minimo global ao longo das epochs
            cost = self.loss_function(y_true, y_pred)
            self.history[epoch] = cost

            if self.verbose:
                print(f' Epoch {epoch}') # acabar aqui


            return self


    def predict(self, dataset: Dataset):
        x = dataset.X 
        for layer in self.layers:
            x = layer.forward(x)
        
        return x


    def cost(self, dataset: Dataset) -> float:
        y_pred = self.predict(dataset)
        return self.loss(dataset.y, y_pred)

    def score(self, dataset: Dataset, scoring_func: Callable = accuracy) -> float:
        y_pred = self.predict(dataset)
        return scoring_func(dataset.y, y_pred)
    
    



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
    print(dataset.to_dataframe())


    w1 = np.array([[20, -20],
                    [20, -20]])

    b1 = np.array([[-30, 10]])


    l1 = Dense(input_size = 2, output_size=2)
    l1.weights = w1
    l1.bias = b1


    w2 = np.array([[20],
                    [20]])

    b2 = np.array([[-10]])

    l2 = Dense(input_size = 2, output_size=1)
    l2.weights = w2
    l2.bias = b2

    l1_sa = SigmoidActivation()
    l2_sa = SigmoidActivation()

    nn_model_sa = NN(layers=[l1, l1_sa, l2, l2_sa])
    nn_model_sa.fit(dataset=dataset)


    print(nn_model_sa.predict(dataset))




    # l1_sma = SoftMaxActivation()
    # l2_sma = SoftMaxActivation()

    # nn_model_sma = NN(layers=[ l1, l1_sma, l2, l2_sma])
    # nn_model_sma.fit(dataset=dataset)

    # l1_rlua = ReLUActivation()
    # l2_rlua = ReLUActivation()

    # nn_model_rlua = NN(layers=[ l1, l1_rlua, l2, l2_rlua])
    # nn_model_rlua.fit(dataset=dataset)



