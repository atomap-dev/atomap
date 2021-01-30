import os
import numpy as np
import atomap.api as am
import atomap.dummy_data as dd
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

my_path = os.path.join(os.path.dirname(__file__), 'working_with_atomic_models')
if not os.path.exists(my_path):
    os.makedirs(my_path)

######
atom_lattice = am.dummy_data.get_perovskite_001_atom_lattice(set_element_info=False)
atom_lattice.set_scale(scale=0.2, units="Å")
atom_lattice.sublattice_list[0].set_element_info('Sr', [0., 4., 8.])
atom_lattice.sublattice_list[1].set_element_info(['O', 'Ti', 'O', 'Ti', 'O'], [0., 2., 4., 6., 8.])
atoms = atom_lattice.convert_to_ase()

atom = atom_lattice.sublattice_list[0].atom_list[45]
atom.set_element_info("La", [0, 4, 8])

atoms = atom_lattice.convert_to_ase()

######
fig, ax = plt.subplots(figsize=(5, 5))
plot_atoms(atoms, ax, radii=0.3, rotation=("7x,7y,0z"))
fig.tight_layout()
fig.savefig(os.path.join(my_path, "perovskite_001_with_la.png"))
