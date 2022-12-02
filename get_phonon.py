import os
kyoto_folder = 'kyoto'
contents = ['BORN','INCAR-force', 'KPOINTS-force', 'PAW_dataset.txt', 'POSCAR-unitcell',
        'disp.yaml', 'INCAR-nac', 'KPOINTS-nac', 'phonon.yaml', 'POSCAR-unitcell.yaml',
        'ORCE_SETS', 'INCAR-relax', 'KPOINTS-relax', 'phonopy.conf', 'phonopy.yaml', 'FORCE_SETS', 'qpoints.yaml']


def get_phonon(idx=1000, k_vec=[0,0,0]):
    """_summary_

    Args:
        idx (int): index of materials project. Defaults to 1000.
            Find id from the database (http://phonondb.mtl.kyoto-u.ac.jp/ph20180417/index.html)
        k_vec (list): [kx, ky, kz] (fractional unit). Defaults to [0,0,0].

    Returns:
        phonon (list): phonon at k_vec.
    """
    os.system(f"wget -P {kyoto_folder} http://phonondb.mtl.kyoto-u.ac.jp/_downloads/mp-{idx}-20180417.tar.lzma")
    os.system(f"rm -r  {kyoto_folder}/{idx}/")
    os.system(f"tar --lzma -xvf {kyoto_folder}/mp-{idx}-20180417.tar.lzma")
    os.system(f"mv mp-{idx}-20180417/  {kyoto_folder}/{idx}")
    os.system(f"rm -r  {kyoto_folder}/mp-{idx}*")
    os.system(f"cp {kyoto_folder}/{idx}/* ./")
    os.system(f"phonopy -p phonopy.conf -c POSCAR-unitcell --qpoints {k_vec[0]} {k_vec[1]} {k_vec[2]}")
    with open("qpoints.yaml", "r") as f:
        d = f.read()
    data = d.replace(' ', '').split("\n")

    phonon = []
    for line in data:
        if line.startswith('freq'):
            phonon.append(float(line[10:]))

    for c in contents:
        if os.path.exists(c):
            os.system(f"rm {c}")
    
    return phonon

if __name__=="__main__":
    phonon = get_phonon(idx=1000, k_vec=[0,0.5,0])
    print(phonon)
    print("Phonon counts: ", len(phonon))
