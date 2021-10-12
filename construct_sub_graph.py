import numpy as np
import torch


class sub_graph:

    def __init__(self, adj, feats, weight=None):
        self.adj = adj
        self.feats = feats
        self.weight = weight

    def _make_mask(self, ids):
        mask = np.zeros(self.adj.shape[0])
        mask[list(ids)] = 1
        mask = np.array(mask, dtype=np.bool)
        return mask

    def _sample(self, ids, sample_num):
        sub_adj = self.adj[ids]
        neighs = list(sub_adj.indices)
        if len(neighs) > sample_num:
            neighs = np.random.choice(neighs, sample_num)
        neighs = set(neighs)
        neighs.add(ids)
        return neighs

    def sample(self, ids, n_layer):
        neighs = set(ids)
        neighs_1 = set(ids)
        for k in range(n_layer):
            tmp_neighs = set()
            for i in neighs:
                i_neighs = self._sample(i, 100)
                if k == 0:
                    neighs_1 = i_neighs | neighs_1
                tmp_neighs = tmp_neighs | i_neighs
            neighs = neighs | tmp_neighs
        return list(neighs), list(neighs_1)

    def _get_data(self, ids):
        mask = self._make_mask(ids)
        adj = self.adj[mask]
        adj = adj.T[mask].T
        feats = self.feats[mask]
        if type(feats) == np.ndarray:
            return adj, feats
        return adj, feats.toarray()

    def construct(self, ids, n_layer):
        neighs, neighs_1 = self.sample(ids, n_layer)
        neighs = sorted(neighs)
        neighs_1 = sorted(neighs_1)
        Mask = []
        for i in range(len(neighs)):
            if neighs[i] in neighs_1:
                Mask.append(i)
        Mask_id = []
        for i in ids:
            Mask_id.append(neighs.index(i))
        adj, feats = self._get_data(neighs)
        adj = adj.toarray()
        adj = adj + np.eye(adj.shape[0])
        adj = np.where(adj > 0, 1, 0)
        raw_adj = adj
        adj = torch.tensor(adj).float()
        if torch.cuda.is_available():
            adj = adj.cuda()
        d = adj.sum(dim=1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D)
        adj = adj.cpu()
        feats = torch.tensor(feats, requires_grad=True).float()
        raw_adj = torch.tensor(raw_adj).float()
        raw_adj = raw_adj[Mask].t()[Mask].t()
        return adj, feats, Mask, raw_adj, neighs, Mask_id
