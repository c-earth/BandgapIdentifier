import glob
import pickle as pkl
from utils.utils_data import gen_cart_qpts, load_raw, get_phonons, gen_qpts_file, read_qpts_file

L = 1000
n = 1000

dat_dir = './data/'
phn_dir = f'{dat_dir}raw/'
dict_file = f'{dat_dir}data_dict.pkl'

data_dict = dict()
cart_qpts = gen_cart_qpts(L, n)
if len(glob.glob(dict_file)) == 0:
    if len(glob.glob(f'{phn_dir}')) == 0:
        load_raw(dat_dir)
    if len(glob.glob('./CART_QPOINTS')) == 0:
        cart_qpts = gen_cart_qpts(L, n)
        gen_qpts_file(cart_qpts, './CART_QPOINTS')
    else:
        cart_qpts = read_qpts_file('./CART_QPOINTS')
    data_dict = get_phonons(phn_dir, cart_qpts)
    with open(dict_file, 'wb') as f:
        pkl.dump(data_dict, f)
else:
    with open(dict_file, 'rb') as f:
        data_dict = pkl.load(f)