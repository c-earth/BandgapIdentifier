import time
import torch
import random
import os
import numpy as np

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
from utils.utils_model import NCELoss, FeatureNetwork, ProjectionNetwork, ContrastiveNetwork, cos_sim, train

# set the model name by time of initialization
run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())

# set hyper-parameters
n = 20                      # grid-sampling frequency
re_sampling = False         # redo grid-sampling

data_ratio = 0.15           # fraction of data from database (10,034) to be used
tr_ratio = 0.8              # fraction of used data for training
k_fold = 5                  # number of cross-validation blocks
max_iter = 100              # total epochs

delta = 0.001               # tolerance for negative samples
phn_mean = 10               # z-normalization shift
phn_sigma = 120             # z-normalization factor
n_layers = [2, 2, 2, 2]     # ResNet hyper-parameters: '[2, 2, 2, 2]' is for ResNet18
n_features = 32             # feature vector space dimensions

batch_size = 1              # always set to 1 to go material by material
lr = 0.005                  # learning rate
weight_decay = 0.05         # AdamW weight decay
schedule_gamma = 0.96       # learning rate decaying factor

# path setting for necessary files and folders
zip_dir = './data/zip/'
raw_dir = './data/raw/'
phn_dir = './data/phn/'
dict_file = f'./data/data_dict_{run_name}.pkl'
model_dir = './models/'

# data preparation
data_dict = load_data(dict_file, phn_dir = phn_dir, raw_dir = raw_dir, zip_dir = zip_dir, data_ratio = data_ratio, re_sampling = re_sampling, n = n)
data_set = build_tr_set(data_dict, delta, phn_mean, phn_sigma)

num = len(data_set)
tr_num = int(num * tr_ratio)
te_num = num - tr_num
fold_size = int(tr_num // k_fold)
tr_nums = [fold_size] * (k_fold - 1) + [tr_num - fold_size * (k_fold - 1)]

tr_set, te_set = torch.utils.data.random_split(data_set, [tr_num, te_num])
tr_sets = torch.utils.data.random_split(tr_set, tr_nums)

# set up loss function, model, and optimization
loss_fn = NCELoss(sim_fn = cos_sim, T = 1)
f_model = FeatureNetwork(n_layers, n_features)
p_model = ProjectionNetwork(n_features)
c_model = ContrastiveNetwork(f_model, p_model)

opt = torch.optim.AdamW(c_model.parameters(), lr = lr, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma = schedule_gamma)

# start training
train(c_model, opt, tr_sets, te_set, loss_fn, run_name, model_dir, max_iter, scheduler, device, batch_size, k_fold)
