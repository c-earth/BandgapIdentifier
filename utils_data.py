import os
import glob
import torch

def load_raw(dat_dir):
    os.system(f'mkdir {dat_dir}zip/; wget -r -nd -A lzma http://phonondb.mtl.kyoto-u.ac.jp/_downloads/ -P {dat_dir}zip/')
    os.system(f'mkdir ./temp/; for f in {dat_dir}zip/*.tar.lzma; do tar --lzma -xvf "$f" -C ./temp/; done')
    os.system(f'mkdir {dat_dir}raw/; mv ./temp/*20180417 {dat_dir}raw/')
    for sub_dir in glob.glob(f'{dat_dir}raw/*/'):
        if sub_dir.endswith('20180417/'):
            os.system(f'mv {sub_dir} {sub_dir[:-10]}')
    os.system(f'rm -r ./temp/')

def get_phonons(phn_dir):
    data_dict = dict()
    for sub_dir in glob.glob(f'{phn_dir}*/'):
        id = os.path.basename(os.path.normpath(sub_dir))
        os.system(f'cp QPOINTS {sub_dir}; cd {sub_dir}; phonopy -p phonopy.conf -c POSCAR-unitcell --qpoints .TRUE.')
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