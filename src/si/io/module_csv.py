import sys
sys.path.insert(0, '/home/monica/Documents/2_ano/sistemas/si/src/si')

# print(sys.path)

from data.dataset import Dataset
import pandas as pd


def read_csv(filename: str, sep: str, features: bool, label: bool):
    # from data.dataset import Dataset
    # utilizar pd.read_csv para retornar o df
    # Tem de reetornar Dataset(df[] que é o x, df[] que é o y, features, label)
    data = pd.read_csv(filename, sep)
    if features and label:  # se tiver nomes nas colunas e a variavel y
        print('TEM FEATURES E Y')
        y = data.iloc[:, -1]
        label = data.columns[-1]
        data = data.iloc[:, :-1]
        features = data.columns
        dataset = Dataset(data, y=y, features=features)

    elif features and label is False:  # se tiver nomes nas colunas mas não tem y
        print('TEM FEATURES')
        features = data.columns
        dataset = Dataset(data, features=features)

    elif features is False and label:  # se tiver y mas não tem nomes nas colunas
        print('TEM Y')
        y = data.iloc[:, -1]
        label = data.columns[-1]
        data = data.iloc[:, :-1]
        dataset = Dataset(data, y=y)

    else:  # quando não tem nem nomes nas colunas nem tem y nos dados
        print('NÃO TEM NADA')
        dataset = Dataset(data)

    return dataset


def write_csv():
    pass


if __name__ == '__main__':
    teste = read_csv('si/datasets/iris.csv', sep=',',
                     features=True, label=True)
    print(teste.X.head(10))


'''
    x = np.array([[1, 2, 3], [3, 1, 3]])
    y = np.array([1, 2])
    features = ['A', 'B', 'C']
    dataset = Dataset(x, y, features, 'as')
'''
