import glob
from utils.utils_data import gen_cart_qpts, gen_qpts_file, read_qpts_file, load_data

L = 1000
n = 1000

dat_dir = './data/'
phn_dir = f'{dat_dir}raw/'
dict_file = f'{dat_dir}data_dict.pkl'
model_dir = './models/'

if len(glob.glob('./CART_QPOINTS')) == 0:
    cart_qpts = gen_cart_qpts(L, n)
    gen_qpts_file(cart_qpts, './CART_QPOINTS')
else:
    cart_qpts = read_qpts_file('./CART_QPOINTS')
data_dict = load_data(dict_file, phn_dir, dat_dir, cart_qpts)
