import torch
import torch.nn as nn
import torch.nn.functional as F
import random

dtype = torch.float64

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
    def __init__(self, in_size, hidden1, hidden2, out_size):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(in_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_size)

    def forward(self, bands):
        x = F.relu(self.fc1(bands))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ProjectionNetwork(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.mlp1 = nn.Linear(n_features, n_features)
        self.mlp2 = nn.Linear(n_features, n_features)
    
    def forward(self, features):
        x = F.relu(self.mlp1(features))
        x = self.mlp2(x)
        return x

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

from torch_geometric.loader import DataLoader
import time
from matplotlib.pyplot import plt
import numpy as np

def train(feature_model,
          projection_model,
          opt,
          tr_set,
          tr_nums,
          te_set,
          loss_fn,
          run_name,
          max_iter,
          scheduler,
          device,
          batch_size,
          k_fold):
    model = CantrastiveNetwork(feature_model, projection_model)
    model.to(device)
    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()

    record_lines = []

    try:
        print('Use model.load_state_dict to load the existing model: ' + run_name + '.torch')
        model.load_state_dict(torch.load(run_name + '.torch')['state'])
    except:
        print('There is no existing model')
        results = {}
        history = []
        s0 = 0
    else:
        print('Use torch.load to load the existing model: ' + run_name + '.torch')
        results = torch.load(run_name + '.torch')
        history = results['history']
        s0 = history[-1]['step'] + 1

    tr_sets = torch.utils.data.random_split(tr_set, tr_nums)
    te_loader = DataLoader(te_set, batch_size = batch_size)
    for step in range(max_iter):
        k = step % k_fold 
        tr_loader = DataLoader(torch.utils.data.ConcatDataset(tr_sets[:k] + tr_sets[k+1:]), batch_size = batch_size, shuffle=True)
        va_loader = DataLoader(tr_sets[k], batch_size = batch_size)
        model.train()
        N = len(tr_loader)
        for i, d in enumerate(tr_loader):
            start = time.time()
            d.to(device)
            batch = model(d)
            loss = loss_fn(batch).cpu()  #! 
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f'num {i+1:4d}/{N}, loss = {loss}, train time = {time.time() - start}', end = '\r')

        end_time = time.time()
        wall = end_time - start_time
        print(wall)
        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step

            valid_avg_loss = evaluate(model, va_loader, loss_fn, device)
            train_avg_loss = evaluate(model, tr_loader, loss_fn, device)
            history.append({
                            'step': s0 + step,
                            'wall': wall,
                            'batch': {
                                    'loss': loss.item(),
                                    },
                            'valid': {
                                    'loss': valid_avg_loss,
                                    },
                            'train': {
                                    'loss': train_avg_loss,
                                    },
                           })
            results = {
                        'history': history,
                        'state': model.state_dict()
                      }
            results_feature = {
                        'state': model.feature_model.state_dict()
                      }
            print(f"Iteration {step+1:4d}   " +
                  f"train loss = {train_avg_loss:8.20f}   " +
                  f"valid loss = {valid_avg_loss:8.20f}   " +
                  f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")

            with open(run_name + '_contrastive.torch', 'wb') as f:
                torch.save(results, f)
            with open(run_name + '_feature.torch', 'wb') as f:
                torch.save(results_feature, f)

            record_line = '%d\t%.20f\t%.20f'%(step,train_avg_loss,valid_avg_loss)
            record_lines.append(record_line)

            loss_plot(run_name, device, './models/' + run_name)
            loss_test_plot(model, device, './models/' + run_name, te_loader, loss_fn)

            # plot the output by the model.feature_model
            
        text_file = open(run_name + ".txt", "w")
        for line in record_lines:
            text_file.write(line + "\n")
        text_file.close()

        if scheduler is not None:
            scheduler.step()


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    loss_cumulative = 0.
    with torch.no_grad():
        for d in dataloader:
            d.to(device)
            batch = model(d)
            loss = loss_fn(batch).cpu()  #!
            loss_cumulative += loss.detach().item()
    return loss_cumulative/len(dataloader)


def loss_plot(model_file, device, fig_file):
    history = torch.load(model_file + '.torch', map_location = device)['history']
    steps = [d['step'] + 1 for d in history]
    loss_train = [d['train']['loss'] for d in history]
    loss_valid = [d['valid']['loss'] for d in history]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(steps, loss_train, 'o-', label='Training')
    ax.plot(steps, loss_valid, 'o-', label='Validation')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.legend()
    fig.savefig(fig_file  + '_loss_train_valid.png')
    plt.close()

def loss_test_plot(model, device, fig_file, dataloader, loss_fn):
    loss_test = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for d in dataloader:
            d.to(device)
            batch = model(d)
            loss = loss_fn(batch).cpu()  #!
            loss_test.append(loss)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(np.array(loss_test), label = 'testing loss: ' + str(np.mean(loss_test)))
    ax.set_ylabel('loss')
    ax.legend()
    fig.savefig(fig_file + '_loss_test.png')
    plt.close()
