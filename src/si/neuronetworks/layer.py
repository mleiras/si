import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np


class Dense:
    def __init__(self, input_size: int = None, output_size: int = None):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(self.input_size, self.output_size) * 0.01
        self.bias = np.zeros((1,self.output_size))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        return np.dot(input_data, self.weights) + self.bias #input_data é uma matriz com colunas==features, as linhas são os exemplos // Para multiplicar matrizes, ao nº de colunas da primeira matriz tem de ser igual ao nº de linhas da segunda matriz (neste caso matriz de pesos)


    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        ...
        #proxima aula
        
        return error 
