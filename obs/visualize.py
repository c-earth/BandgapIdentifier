import matplotlib.pyplot as plt
from pymatgen.core.lattice import Lattice
from pymatgen.electronic_structure.plotter import plot_lattice_vectors, plot_points, plot_wigner_seitz

fontsize = 20

def plot_ks(ddb_lat, ks):
    fig = plt.figure(figsize = (24, 24))
    ax1 = fig.add_subplot(projection = '3d')

    ax1.set_title('k points', fontsize = fontsize)
    plot_lattice_vectors(ddb_lat, ax = ax1)
    plot_points(ks, ddb_lat, ax = ax1)
    plot_wigner_seitz(ddb_lat, ax = ax1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    plt.show()

plot_ks(Lattice.from_parameters(1, 1, 1, 60, 60, 60), [[0.5, 0.5, 0.5], [0, 0, 0]])
