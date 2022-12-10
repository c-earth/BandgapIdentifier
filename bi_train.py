import glob
import torch
import time
from utils.utils_data import gen_cart_qpts, gen_qpts_file, read_qpts_file, load_data
from utils.utils_model import BandAugmentations, NCELoss, FeatureNetwork, ProjectionNetwork, ContrastiveNetwork, cos_sim, train
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

L = 1000
n = 1000

dat_dir = './data/'
phn_dir = f'{dat_dir}raw/'
dict_file = f'{dat_dir}data_dict.pkl'
model_dir = './models/'
run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
max_iter = 10
n_layers = 5
n_features = 512
tr_ratio = 0.9
batch_size = 1
k_fold = 5

if len(glob.glob('./CART_QPOINTS')) == 0:
    cart_qpts = gen_cart_qpts(L, n)
    gen_qpts_file(cart_qpts, './CART_QPOINTS')
else:
    cart_qpts = read_qpts_file('./CART_QPOINTS')
data_dict = load_data(dict_file, phn_dir, dat_dir, cart_qpts)

# ToDo: Use BandAugmentations()

num = len(data_dict)
tr_nums = [int((num * tr_ratio)//k_fold)] * k_fold
te_num = num - sum(tr_nums)
tr_set, te_set = torch.utils.data.random_split(list(data_dict.values()), [num - te_num, te_num])

loss_fn = NCELoss(sim_fn=cos_sim, T=1)
lr = 0.005
weight_decay = 0.05
schedule_gamma = 0.96

f_model = FeatureNetwork(n_layers, 
                         num_classes = 1000, 
                         zero_init_residual = False, 
                         groups =1, 
                         width_per_group = 64, 
                         replace_stride_with_dilation = None, 
                         norm_layer = None)
p_model = ProjectionNetwork(n_features)
c_model = ContrastiveNetwork(f_model, p_model)

opt = torch.optim.AdamW(c_model.parameters(), lr = lr, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma = schedule_gamma)

train(c_model,
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
          k_fold)