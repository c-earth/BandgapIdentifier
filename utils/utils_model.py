import torch
import torch.nn as nn
import torch.nn.functional as F
import random

dtype = torch.float64
n_features = 2048

class BandAugmentations():
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, bands):
        pivot = [list(range(len(bands))),[random.randint(0, bands.shape[-1]-1)for i in range(len(bands))]]
        inc_aug = bands + torch.abs(torch.normal(torch.zeros(bands.shape), self.delta*torch.ones(bands.shape)))
        dec_aug = bands - torch.abs(torch.normal(torch.zeros(bands.shape), self.delta*torch.ones(bands.shape)))
        inc_aug[pivot] = bands[pivot]
        dec_aug[pivot] = bands[pivot]
        return torch.concat([inc_aug, dec_aug], dim = 0)

class NCELoss(torch.nn.Module):
    def __init__(self, sim_fn, T = 1):
        super(NCELoss, self).__init__()
        self.sim_fn = sim_fn
        self.T = T

    def forward(self, features, not_neg_mask, pos_mask):
        sims = torch.masked_fill(self.sim_fn(features) / self.T, not_neg_mask, -1e10)
        return torch.mean(-sims[pos_mask] + torch.logsumexp(sims, dim = -1))

class FeatureNetwork(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, bands):
        pass

class ProjectionNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Linear(n_features, n_features)
        self.mlp2 = nn.Linear(n_features, n_features)
    
    def forward(self, features):
        features = F.relu(self.mlp1(features))
        return self.mlp2(features)

class ContrastiveNetwork(torch.nn.Module):
    def __init__(self, feature_model, projection_model):
        super().__init__()
        self.feature_model = feature_model
        self.projection_model = projection_model
    
    def forward(self, data):
        features = self.feature_model(data.bands)
        features = self.projection_model(features)
        return features

def cos_sim(features):
    nfeatures = torch.nn.functional.normalize(features, p = 2, dim = 1)
    return torch.matmul(nfeatures, nfeatures.T)
    
def train():
    pass
