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

def train():
    pass
