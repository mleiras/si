import numpy as np
import pandas as pd
from typing import Tuple


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray=None, features=None, label: str=None):
        
        if X is None:
            raise ValueError("X must not be None")
        
        if features is None or features is False:
            features = [str(i) for i in range(X.shape[1])]
        else:
            features = list(features)

        if y is not None and label is None:
            label = 'y'
        
        self.X = X
        self.y = y  # np array de 1 dimensão
        self.label = label  # string
        self.features = features


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
        else:
            raise ValueError("Dataset does not have y")

    def get_mean(self):
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

    def dropna(self):
        '''
        Remove todas as amostras que contêm pelo menos um valor nulo (NaN).
        '''
        self.X = self.X[~np.isnan(self.X).any(axis=1)]
        return self.X


    def fillna(self, num: int) -> None:
        '''Substitui todas os valores nulos por outro valor (argumento da função/método).

        Parameters
        ----------
        num : int
            Valor de substituição
        '''
        self.X = np.nan_to_num(self.X, nan=num)

    def from_random(n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features = None,
                    label: str = None):
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return Dataset(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df
            

if __name__ == '__main__':
    x = np.array([[1, 2, 3], [3, 1, 3]])
    y = np.array([5, 5])
    features = ['A', 'B', 'C']
    dataset = Dataset(x, y, features, 'as')
    print('X:\n', dataset.X)
    print('y:\n', dataset.y)
    print('X shape:\n', dataset.shape())
    print('Label:\n', dataset.has_label())
    print('Classes:\n', dataset.get_classes())
    print('Mean:\n', dataset.get_mean())
    print('Variance:\n', dataset.get_variance())
    print('Median:\n', dataset.get_median())
    print('Min:\n', dataset.get_min())
    print('Max:\n', dataset.get_max())
    print('Summary:\n', dataset.summary())
    print()
    x2 = np.array([[1, 2, 3], [3, 1, np.nan]])
    dataset2 = Dataset(x2, y, features, 'label')
    print('Dataset 2:\n', dataset2.X)
    dataset2.dropna()
    print('Dataset 2 depois de dropna()):\n', dataset2.X)
    dataset2 = Dataset(x2, y, features, 'label')
    dataset2.fillna(100)
    print('Dataset 2 depois de fillna(100):\n', dataset2.X)


