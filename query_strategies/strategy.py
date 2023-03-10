import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Strategy:
    def __init__(self, dataset, net):
        self.dataset = dataset
        self.net = net
    def query(self, n):
        pass
    
    def get_embeddings(self, data):
        embed=self.net.embed
        features = []
        for i_batch, datum in enumerate(data): 
            img = datum[0].float()
            output = embed(img)
            features.append(output)
        features = torch.cat(features, dim=0).detach()
        return features

