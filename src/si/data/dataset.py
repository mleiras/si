import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, X, y=None, features=None, label=None) -> None:
        self.X = X  # np array
        self.y = y  # np array de 1 dimensão
        self.features = features  # pode ser uma lista - atributos em português
        self.label = label  # string

    def shape(self):
        return self.X.shape

    def has_label(self):  # se existir vetor y devolve True
        if self.y is not None:
            return True
        else:
            return False

    def get_classes(self):  # valores unicos do vetor y (se existir)
        if self.has_label():
            return np.unique(self.y)

    def get_mean(self):
        # se não colocar axis, vai fazer a média da matriz toda
        return np.mean(self.X, axis=0)

    def get_variance(self):
        return np.var(self.X, axis=0)

    def get_median(self):
        return np.median(self.X, axis=0)

    def get_min(self):
        return np.min(self.X, axis=0)

    def get_max(self):
        return np.max(self.X, axis=0)

    def summary(self):
        return pd.DataFrame(
            {'Média': self.get_mean(),
             'Variância': self.get_variance(),
             'Mediana': self.get_median(),
             'Mínimo': self.get_min(),
             'Máximo': self.get_max(), }
        )


if __name__ == '__main__':
    x = np.array([[1, 2, 3], [3, 1, 3]])
    y = np.array([1, 2])
    features = ['A', 'B', 'C']
    dataset = Dataset(x, y, features, 'as')
    print(dataset.X)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())
