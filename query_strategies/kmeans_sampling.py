from .strategy import Strategy
from fast_pytorch_kmeans import KMeans
import torch
class KMeansSampling(Strategy):
    def __init__(self, dataset, net):
        super(KMeansSampling, self).__init__(dataset, net)

    def euclidean_dist(self,x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def query(self, c,n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_class_data(c)
        embeddings = self.get_embeddings(unlabeled_data)
        kmeans = KMeans(n_clusters=n, mode='euclidean', verbose=1)
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids
        dist_matrix = self.euclidean_dist(centers, embeddings)
        q_idxs = unlabeled_idxs[torch.argmin(dist_matrix,dim=1)]
        return q_idxs

    def query_match_sample(self, c,n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_class_data(c)
        embeddings = self.get_embeddings(unlabeled_data)
        kmeans = KMeans(n_clusters=n, mode='euclidean')
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids
        dist_matrix = self.euclidean_dist(centers, embeddings)
        q_idxs = unlabeled_idxs[torch.argmin(dist_matrix,dim=1)]
        return q_idxs