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
    def __init__(self, k: int = 3):
        self.k = k
        self.k_mers = []

    
    def fit(self, dataset):
        for combination in itertools.product('ACTG', repeat=self.k):
            self.k_mers.append(''.join(combination))
        return self

    def _get_kmer(self, seq):
        dicio = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(seq)-self.k +1):
            dicio[seq[i:i+self.k]]+= 1

        return np.array([dicio[k_mer]/len(seq) for k_mer in self.k_mers])


    def transform(self, dataset):
        sequences_kmer = [self._get_kmer(seq) for seq in dataset.X.iloc[:,0]]
        sequences_kmer = np.array(sequences_kmer)

        return Dataset(X=sequences_kmer, y=dataset.y, features=list(self.k_mers), label=dataset.label)
        

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)
        


if __name__ == '__main__':
    from io_folder.module_csv import read_csv
    tfbs_dataset = read_csv('/home/monica/Documents/2_ano/sistemas/si/datasets/tfbs.csv', sep=',',features=True, label=True)
    kmer = KMer(3)
    kmer_dataset = kmer.fit_transform(tfbs_dataset)
    print(kmer_dataset.features)
