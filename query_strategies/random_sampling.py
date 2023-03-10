import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, dataset, net):
        super(RandomSampling, self).__init__(dataset, net)

    def query(self, c,n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(c)
        return np.random.choice(unlabeled_idxs, n, replace=False)
