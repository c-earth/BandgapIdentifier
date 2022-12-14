import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader

# set deterministic behavior before importing other sub-modules
all_seed = 177013
set_deterministic = True

print(f'Deterministic behavior is set to be {set_deterministic}')
if set_deterministic:
    np.random.seed(all_seed)
    random.seed(all_seed)
    torch.manual_seed(all_seed)
    torch.cuda.manual_seed_all(all_seed)
    os.environ['PYTHONHASHSEED'] = str(all_seed)
    print(f'Random seed is set as {all_seed}')

# set torch module behavior before importing other sub-modules
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('device: ', device)
torch.set_default_dtype(torch.float64)

from utils.utils_data import load_data, build_tr_set
from utils.utils_model import FeatureNetwork, ProjectionNetwork, ContrastiveNetwork
from utils.utils_cluster import auto_fit_gmm

# set the model name
run_name = '221212-104840'

# set hyper-parameters
delta = 0.01                # tolerance for negative samples
phn_mean = 10               # z-normalization shift
phn_sigma = 120             # z-normalization factor
n_layers = [2, 2, 2, 2]     # ResNet hyper-parameters: '[2, 2, 2, 2]' is for ResNet18
n_features = 16             # feature vector space dimensions

batch_size = 1              # always set to 1 to go material by material

# path setting for necessary files and folders
dict_file = f'./data/data_dict_{run_name}.pkl'
model_dir = './models/'
model_file = f'{model_dir}{run_name}_contrastive.torch'

# set up and load model
f_model = FeatureNetwork(n_layers, n_features)
p_model = ProjectionNetwork(n_features)
c_model = ContrastiveNetwork(f_model, p_model)
f_model.to(device)
f_model.load_state_dict(torch.load(model_file)['state'])

# load input data
data_dict = load_data(dict_file)
data_set = build_tr_set(data_dict, delta, phn_mean, phn_sigma)
data_loader = DataLoader(data_set, batch_size = batch_size)

# predict phonon band grouping
for i, d in enumerate(data_loader):
    print(d.id)
    d.to(device)
    features = c_model(d.phns)
    band_groups, bic = auto_fit_gmm(features.cpu().detach().numpy())
    print(band_groups)
    plt.figure()
    plt.plot(bic)
    plt.savefig(f'{d.id}_BIC.png')
