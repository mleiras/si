import numpy as np

class VarianceThreshold:
    def __init__(self, threshold):
        self.threshold = threshold # linha de corte/valor de corte
        self.variance = None

    def fit(self, dataset):
        variance = np.var(dataset.X) ## ou então Dataset.get_var
        self.variance = variance
        return self # retorna ele próprio


    def transform(self, dataset):
        mask = self.variance > self.threshold
        novo_X = dataset.X[:,mask]
        features= np.array(dataset.features)[mask] # selecionar as features daquelas que tem variance > threshold
        return Dataset(novo_X, u=dataset.y, features=list(features), label=dataset.label)



    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)



if __name__ == '__main__':
    from si.data.dataset import Dataset



    # dataset = Dataset(
    #     X = np.array([[0,2,0,3],
    #     ])
    # )
    # dataset.X[:,0] = 0



    # selector = VarianceThreshold(threshold=0.1)
    # selector = selector.fit(dataset)



    

