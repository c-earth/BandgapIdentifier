import torch
import random
import os
import glob
import numpy as np
import pickle as pkl

from torch_geometric.data import Data
from scipy.spatial.distance import pdist, squareform

def load_raw(zip_dir, raw_dir):
    '''
    load data from database
    arguments:
        zip_dir: folder for zip files
        raw_dir: folder for extracted data files
    return:
        None
    '''
    # download and extract data
    os.system(f'mkdir {zip_dir}; wget -r -nd -A lzma http://phonondb.mtl.kyoto-u.ac.jp/_downloads/ -P {zip_dir}')
    os.system(f'mkdir ./temp/; for f in {zip_dir}*.tar.lzma; do tar --lzma -xvf "$f" -C ./temp/; done')
    os.system(f'mkdir {raw_dir}; mv ./temp/*20180417 {raw_dir}')

    # manage folder names
    for sub_dir in glob.glob(f'{raw_dir}*/'):
        if sub_dir.endswith('20180417/'):
            os.system(f'mv {sub_dir} {sub_dir[:-10]}')
    os.system(f'rm -r ./temp/')

def gen_qpts(n):
    '''
    generate uniform grid-sampling of wave vectors
    arguments:
        n: grid-sampling frequency
    return:
        (n, n, n, 3) matrix of wave vectors at different grid points
    '''
    return (np.array([[[[i, j, k] for i in range(n)] for j in range(n)] for k in range(n)]) + np.random.uniform(0.0, 1.0, (n, n, n, 3))) / n

def gen_qpts_file(qpts, file_name):
    '''
    generate Phonopy readable file for all wave vectors
    arguments:
        qpts: list of wave vectors in unit of primitive reciprocal basis of the material
        file_name: wave vector file name
    return:
        None
    '''
    data = f'{len(qpts)}\n'
    for qpt in qpts:
        data += ' '.join(map(str, qpt)) + '\n'
    with open(file_name, 'w') as f:
        data = f.write(data)

def gen_phonons(raw_dir, phn_dir, n, qpts_name = 'QPOINTS'):
    '''
    use Phonopy to generate phonon data
    arguments:
        raw_dir: folder for extracted files
        phn_dir: folder for phonon data files
        n: grid-sampling frequency
        qpts_name: wave vector file name
    return:
        None
    '''
    N = len(glob.glob(f'{raw_dir}*/'))
    c = 0
    failed_ids = []

    for sub_dir in glob.glob(f'{raw_dir}*/'):
        # print progress and material's id
        c += 1
        print(c,'/',N)
        id = os.path.basename(os.path.normpath(sub_dir))
        print(id)

        try:
            # grid-sampling wave vectors
            qpts = gen_qpts(n)
            fqpts = qpts.reshape(-1, 3)
            gen_qpts_file(fqpts, qpts_name)

            # call Phonopy to calculate phonon frequency response data at sampled wave vectors
            os.system(f'mv {qpts_name} {sub_dir}QPOINTS; cd {sub_dir}; phonopy -p phonopy.conf -c POSCAR-unitcell --qpoints .TRUE.')

            # make phonon frequency response list from Phonopy output file
            with open(f'{sub_dir}qpoints.yaml', 'r') as f:
                data = f.readlines()
            phonons = []
            phonon = None
            for line in data:
                cline = line.replace('- ', '').replace(' ', '').strip('\n').split(':')
                if cline[0] == 'q-position':
                    if phonon:
                        phonons.append(phonon)
                    phonon = []
                elif cline[0] == 'frequency':
                    phonon.append(float(cline[1]))
            phonons.append(phonon)
            phonons = np.array(phonons)

            # calculate minimum frequency difference between every phonon bands
            min_dist = squareform(pdist(np.transpose(phonons), lambda u, v: np.min(np.abs(u-v))))

            # make phonon frequency response object
            phonons = phonons.reshape(n, n, n, -1)

            # save data
            with open(f'{phn_dir}{n}/{id}.pkl', 'wb') as f:
                pkl.dump({'id': id, 'n': n, 'qpts': qpts, 'phn': phonons, 'md': min_dist}, f)
        except:
            failed_ids.append(id)

    # print list of all materials' ids that Phonopy cannot generate phonon data
    print(failed_ids)

def get_phonons(phn_dir, data_ratio, n):
    '''
    get phonon data generated from Phonopy
    arguments:
        phn_dir: folder for phonon data files
        data_ratio: fraction of data from database (10,034) to be used
        n: grid-sampling frequency
    return:
        dictionary that link material's id to its phonon data, mean, and sd of all phonon responses in the data
    '''
    data_dict = dict()
    for file in glob.glob(f'{phn_dir}{n}/*'):
        if random.random() < data_ratio:
            with open(file, 'rb') as f:
                d = pkl.load(f)
            data_dict[d['id']] = d
    return data_dict

def load_data(dict_file, phn_dir = None, raw_dir = None, zip_dir = None, data_ratio = 0.1, re_sampling = False, n = 20):
    '''
    load existing phonon data dictionary, regenerate phonon data with Phonopy if specify,
    and download data from database if needed
    arguments:
        dict_file: file for saving data dictionary
        phn_dir: folder for phonon data files
        raw_dir: folder for extracted files
        zip_dir: folder for zip files
        data_ratio: fraction of data from database (10,034) to be used
        re_sampling: regenerate phonon data director
        n: grid-sampling frequency
    return:
        dictionary that link material's id to its phonon data
    '''
    if len(glob.glob(dict_file)) == 0:
        if re_sampling:
            if len(glob.glob(raw_dir)) == 0: 
                load_raw(zip_dir, raw_dir)

            gen_phonons(raw_dir, phn_dir, n)

        data_dict = get_phonons(phn_dir, data_ratio, n)

        with open(dict_file, 'wb') as f:
            pkl.dump(data_dict, f)

    else:
        with open(dict_file, 'rb') as f:
            data_dict = pkl.load(f)

    return data_dict

def build_tr_set(data_dict, delta, phn_mean, phn_sigma):
    '''
    generate torch data object set from data dictionary
    arguments:
        data_dict: dictionary that link material's id to its phonon data
        delta: tolerance for negative samples
        phn_mean: phonon z-normalization shift
        phn_sigma: phonon z-normalization factor
    return:
        list of torch data object containing materials' basic data, and phonon frequency responses data
        and other data used for training
    '''
    tr_set = []
    for d in data_dict.values():

        # get information from data dictionary
        id = d['id']        # material id
        n = d['n']          # grid-sampling frequency
        qpts = d['qpts']    # wave vectors at each grid point
        phns = d['phn']     # phonon frequency responses at each grid point
        md = d['md']        # minimum frequency difference between every phonon bands

        # z-normalize phonon frequency response
        phns = (phns - phn_mean)/phn_sigma

        # mask for non-negative sample pairs of bands with respect to the original material's bands
        non_neg_mask = md <= delta
        non_neg_mask = np.hstack((non_neg_mask, (1 - np.eye(len(non_neg_mask))).astype(non_neg_mask.dtype)))
        non_neg_mask = np.vstack((non_neg_mask, np.ones(non_neg_mask.shape, dtype = non_neg_mask.dtype)))

        # mask for positive sample pairs of bands
        self_mask = torch.eye(2 * phns.shape[-1], dtype = torch.bool)
        pos_mask = self_mask.roll(shifts = phns.shape[-1], dims=0)

        # generate torch data object
        data = Data(id = id,
                    n = n,
                    qpts = torch.from_numpy(qpts),
                    phns = torch.permute(torch.from_numpy(np.expand_dims(phns, axis = 0)), 
                                         (4, 0, 1, 2, 3)), # reshape phonon data object to match torch convolution input criteria
                    non_neg_mask = torch.from_numpy(non_neg_mask),
                    pos_mask = pos_mask)

        tr_set.append(data)
    return tr_set
