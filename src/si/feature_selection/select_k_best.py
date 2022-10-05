


class SelectKBest:
    def __init__(self, score_func, k: int):
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def fit(self, dataset):
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset):
        idx = np.argsort(self.F)[-self.k:]
        features= np.array(dataset.features)[idx] # selecionar as features com base nos idx 
        return Dataset(dataset.X[:,idx], u=dataset.y, features=list(features), label=dataset.label)


    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)



