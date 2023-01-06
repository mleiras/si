import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from data.dataset import Dataset
import itertools

class KMer:
    def __init__(self, k: int = 3, alphabet: str = 'DNA'):
        self.k = k
        self.alphabet = alphabet.upper()

        if self.alphabet == 'DNA':
            self.alphabet = 'ACTG'
        elif self.alphabet == 'PROTEIN':
            self.alphabet = 'FLIMVSPTAY_HQNKDECWRG'
        else:
            self.alphabet = self.alphabet
        
        self.k_mers = None

    
    def fit(self, dataset):
        self.k_mers = [''.join(k_mer) for k_mer in itertools.product(self.alphabet, repeat=self.k)]
        return self


    def _get_kmer(self, seq):
        dicio = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(seq)-self.k +1):
            dicio[seq[i:i+self.k]]+= 1

        return np.array([dicio[k_mer]/len(seq) for k_mer in self.k_mers])


    def transform(self, dataset):
        sequences_kmer = [self._get_kmer(seq) for seq in dataset.X[:,0]]
        sequences_kmer = np.array(sequences_kmer)

        return Dataset(X=sequences_kmer, y=dataset.y, features=self.k_mers, label=dataset.label)
        

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)
        


if __name__ == '__main__':
    dataset_ = Dataset(X=np.array([['ACTGTTTAGCGGA', 'ACTGTTTAGCGGA']]),
                       y=np.array([1, 0]),
                       features=['sequence'],
                       label='label')

    k_mer_ = KMer(k=2)
    dataset_ = k_mer_.fit_transform(dataset_)
    print(dataset_.X)
    print(dataset_.features)

    print('---------------------------------------------------------')
    
    from io_folder.module_csv import read_csv
    from sklearn.preprocessing import StandardScaler
    from linear_model.logistic_regression import LogisticRegression
    from model_selection.split import train_test_split


    tfbs_dataset = read_csv('/home/monica/Documents/2_ano/sistemas/si/datasets/tfbs.csv', sep=',',features=True, label=True)
    kmer = KMer(3)
    kmer_dataset = kmer.fit_transform(tfbs_dataset)
    print(kmer_dataset.features)

    kmer_dataset.X = StandardScaler().fit_transform(kmer_dataset.X)
    kmer_dataset_train, kmer_dataset_test = train_test_split(kmer_dataset, test_size=0.2)
    model = LogisticRegression()
    model.fit(kmer_dataset_train)
    score = model.score(kmer_dataset_test)
    print(f"Score: {score}")

    # model.cost_plot()
