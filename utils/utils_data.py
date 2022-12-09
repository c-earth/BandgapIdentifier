import os
import glob
import torch
import numpy as np
from mp_api.client import MPRester
import pickle as pkl

api_key = 'KKYDb56RGe2YQhuLJqE17k7AztkJ6Fyj'

def load_raw(dat_dir):
    os.system(f'mkdir {dat_dir}zip/; wget -r -nd -A lzma http://phonondb.mtl.kyoto-u.ac.jp/_downloads/ -P {dat_dir}zip/')
    os.system(f'mkdir ./temp/; for f in {dat_dir}zip/*.tar.lzma; do tar --lzma -xvf "$f" -C ./temp/; done')
    os.system(f'mkdir {dat_dir}raw/; mv ./temp/*20180417 {dat_dir}raw/')
    for sub_dir in glob.glob(f'{dat_dir}raw/*/'):
        if sub_dir.endswith('20180417/'):
            os.system(f'mv {sub_dir} {sub_dir[:-10]}')
    os.system(f'rm -r ./temp/')

def get_reciprocal_lattice(id):
    with MPRester(api_key) as mpr:
        struct = mpr.get_structure_by_material_id(id)
    return struct.lattice.reciprocal_lattice

def get_phonons(phn_dir, cart_qpts):
    data_dict = dict()
    for sub_dir in glob.glob(f'{phn_dir}*/'):
        id = os.path.basename(os.path.normpath(sub_dir))
        lattice = get_reciprocal_lattice(id)
        qpts = lattice.get_fractional_coords(cart_qpts)
        gen_qpts_file(qpts, 'QPOINTS')
        os.system(f'mv QPOINTS {sub_dir}; cd {sub_dir}; phonopy -p phonopy.conf -c POSCAR-unitcell --qpoints .TRUE.')
        with open(f'{sub_dir}qpoints.yaml', 'r') as f:
            data = f.readlines()
        qpts = []
        phonons = []
        phonon = None
        for line in data:
            cline = line.replace('- ', '').replace(' ', '').strip('\n').split(':')
            if cline[0] == 'q-position':
                qpts.append(eval(cline[1]))
                if phonon:
                    phonons.append(phonon)
                phonon = []
            elif cline[0] == 'frequency':
                phonon.append(float(cline[1]))
        phonons.append(phonon)
        data_dict[id] = [torch.tensor(qpts), torch.tensor(phonons)]
    return data_dict

def gen_qpts_file(qpts, file_name):
    data = f'{len(qpts)}\n'
    for qpt in qpts:
        data += ' '.join(map(str, qpt)) + '\n'
    with open(file_name, 'w') as f:
        data = f.write(data)

def read_qpts_file(file_name):
    with open(file_name, 'r') as f:
        data = f.readlines()
    qpts = []
    for line in data:
        cline = line.strip('\n').split()
        if len(cline) == 3:
            qpts.append([eval(c) for c in cline])
    return qpts

def gen_cart_qpts(L, n):
    return L * np.random.uniform(0.0, 1.0, 3*n).reshape((-1, 3))

def load_data(dict_file, phn_dir, dat_dir, cart_qpts):
    if len(glob.glob(dict_file)) == 0:
        if len(glob.glob(f'{phn_dir}')) == 0:
            load_raw(dat_dir)
        data_dict = get_phonons(phn_dir, cart_qpts)
        with open(dict_file, 'wb') as f:
            pkl.dump(data_dict, f)
    else:
        with open(dict_file, 'rb') as f:
            data_dict = pkl.load(f)
    return data_dict
