
#%%
"""
python==3.10.0
phonopy==2.15.1 (setup: https://phonopy.github.io/phonopy/install.html)
"""
import glob
import os
import torch
kyoto_folder = 'kyoto'  # Folder to store the in[put files
# qpoints_dir = 'ky_qpts'   # Folder to store the output of phonopy here # Folder to keep the phonon output file. 
contents = ['BORN','INCAR-force', 'KPOINTS-force', 'PAW_dataset.txt', 'POSCAR-unitcell',
        'disp.yaml', 'INCAR-nac', 'KPOINTS-nac', 'phonon.yaml', 'POSCAR-unitcell.yaml',
        'ORCE_SETS', 'INCAR-relax', 'KPOINTS-relax', 'phonopy.conf', 'phonopy.yaml', 'FORCE_SETS', 'qpoints.yaml']


def get_phonons(idx=1000, k_vecs=torch.Tensor([[0,0,0]])):
    """_summary_

    Args:
        idx (int): index of materials project. Defaults to 1000.
            Find id from the database (http://phonondb.mtl.kyoto-u.ac.jp/ph20180417/index.html)
        k_vecs (torch.Tensor): [[ka0, kb0, kc0], [ka1, kb1, kc1], ...] (fractional unit). Defaults to torch.Tensor([[0,0,0]]).

    Returns:
        phonon (torch.Tensor): phonon at each k_vec points.
                                shape=(#k_vecs.shape[0], #phonons)
    """
    # if len(glob.glob(f'{kyoto_folder}/{idx}/')) == 0:
    os.system(f"wget -P {kyoto_folder} http://phonondb.mtl.kyoto-u.ac.jp/_downloads/mp-{idx}-20180417.tar.lzma")
    os.system(f"rm -r  {kyoto_folder}/{idx}/")
    os.system(f"tar --lzma -xvf {kyoto_folder}/mp-{idx}-20180417.tar.lzma")
    os.system(f"mv mp-{idx}-20180417/  {kyoto_folder}/{idx}")
    os.system(f"rm -r  {kyoto_folder}/mp-{idx}*")
    os.system(f"cp {kyoto_folder}/{idx}/* ./")
    phonons = []
    for l in range(k_vecs.shape[0]):
        os.system(f"phonopy -p phonopy.conf -c POSCAR-unitcell --qpoints {k_vecs[l, 0].item()} {k_vecs[l, 1].item()} {k_vecs[l, 2].item()}")
        # os.system(f"mv qpoints.yaml {qpoints_dir}/qpoints_{idx}.yaml")
        # my_file = open(f"{qpoints_dir}/qpoints_{idx}.yaml", "r")
        with open("qpoints.yaml", "r") as f:
            d = f.read()
        data = d.replace(' ', '').split("\n")

        phonon = []
        for line in data:
            if line.startswith('freq'):
                phonon.append(float(line[10:]))
        phonons.append(phonon)

    for c in contents:
        if os.path.exists(c):
            os.system(f"rm {c}")
    
    return torch.Tensor(phonons)

if __name__=="__main__":
    k_nums = [50, 50, 50]
    k_mins = [0, 0, 0]
    k_maxs = [1, 1, 1]
    k_num_all = k_nums[0]*k_nums[1]*k_nums[2]
    k_vecs_in = torch.zeros(k_num_all, 3)
    count = 0
    for ia in range(k_nums[0]):
        ka = k_mins[0] + (k_maxs[0]-k_mins[0])*ia/k_nums[0]
        for ib in range(k_nums[1]):
            kb = k_mins[1] + (k_maxs[1]-k_mins[1])*ib/k_nums[1]
            for ic in range(k_nums[2]):
                kc = k_mins[2] + (k_maxs[2]-k_mins[2])*ic/k_nums[2]
                k_vecs_in[count, :] = torch.Tensor([ka, kb, kc])
                count+=1
    
    phonons = get_phonons(idx=1000, k_vecs=k_vecs_in)
    print("---k_vecs_in---")
    print(k_vecs_in)
    print(k_vecs_in.shape)
    print("---phonons---")
    print(phonons)
    print(phonons.shape)
    # bands = []
    # k_num = 10
    # k_vecs_in = torch.zeros(k_num, 3)
    # count = 0
    # for i in range(0, 100, k_num):
    #     k_vecs_in[count, :] = torch.Tensor([i, i, i])
    #     count+=1
    # phonons = get_phonons(idx=1000, k_vecs=k_vecs_in)
    # print("---k_vecs_in---")
    # print(k_vecs_in)
    # print(phonons)

