import os
import atomap.api as am
from atomap.tools import remove_atoms_from_image_using_2d_gaussian

my_path = os.path.join(os.path.dirname(__file__), 'plotting_tutorial')
if not os.path.exists(my_path):
    os.makedirs(my_path)


def plot_fantasite(s):
    s.plot()
    s._plot.signal_plot.figure.savefig(
            os.path.join(my_path, 'fantasite.png'))


def plot_atom_lattice(atom_lattice):
    s = atom_lattice.get_sublattice_atom_list_on_image()
    s.plot()
    s._plot.signal_plot.figure.savefig(
            os.path.join(my_path, 'atom_lattice.png'))


s = am.dummy_data.get_fantasite()
A_positions = am.get_atom_positions(s, separation=12, pca=True)
sublattice_A = am.Sublattice(A_positions, image=s.data, color='r', name='A')
sublattice_A.find_nearest_neighbors()
sublattice_A.refine_atom_positions_using_center_of_mass()
sublattice_A.refine_atom_positions_using_2d_gaussian()
sublattice_A.construct_zone_axes()

direction_001 = sublattice_A.zones_axis_average_distances[1]
B_positions = sublattice_A.find_missing_atoms_from_zone_vector(direction_001)
image_without_A = remove_atoms_from_image_using_2d_gaussian(
        sublattice_A.image, sublattice_A)

sublattice_B = am.Sublattice(
        B_positions, image_without_A, color='blue', name='B')
sublattice_B.construct_zone_axes()
sublattice_B.refine_atom_positions_using_center_of_mass()
sublattice_B.refine_atom_positions_using_2d_gaussian()
atom_lattice = am.Atom_Lattice(
        image=s.data, name='fantasite',
        sublattice_list=[sublattice_A, sublattice_B])
atom_lattice.save(os.path.join(my_path, "fantasite.hdf5"))

plot_fantasite(s)
plot_atom_lattice(atom_lattice)
