import torch

dtype = torch.float64

class BandAugmentations():
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, bands):
        pivot = torch.randint(0, bands.shape[-1], (bands.shape[0], 1), device = bands.device)
        inc_aug = bands + torch.abs(bands.normal(bands.zeros(), self.delta*bands.ones()))
        dec_aug = bands - torch.abs(bands.normal(bands.zeros(), self.delta*bands.ones()))
        inc_aug[pivot] = bands[pivot]
        dec_aug[pivot] = bands[pivot]
        return torch.concat([inc_aug, dec_aug], dim = 0)

class NCELoss(torch.nn.Module):
    def __init__(self, sim_fn, T = 1):
        self.sim_fn = sim_fn
        self.T = T

    def forward(self, batch, not_neg_mask, pos_mask):
        sims = torch.masked_fill(self.sim_fn(batch) / self.T, not_neg_mask, -1e10)
        return torch.mean(-sims[pos_mask] + torch.logsumexp(sims, dim = -1))

class FeatureNetwork(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, bands):
        pass

class ProjectionNetwork(torch.nn.Module):
    def __init__(self):
        pass
    
    def forward(self, features):
        pass

class CantrastiveNetwork(torch.nn.Module):
    def __init__(self, feature_model, projection_model):
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
