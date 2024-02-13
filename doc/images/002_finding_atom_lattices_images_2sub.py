import os
import hyperspy.api as hs
import atomap.api as am
import atomap.dummy_data as dummy_data
from atomap.tools import remove_atoms_from_image_using_2d_gaussian

my_path = os.path.join(os.path.dirname(__file__), 'finding_atom_lattices')
if not os.path.exists(my_path):
    os.makedirs(my_path)


def save_sub_A_image(sublattice):
    s = sublattice.get_atom_list_on_image()
    s.plot()
    s._plot.signal_plot.figure.savefig(
            os.path.join(my_path, 'sublattice_A.png'))


def plot_planes_figure(sublattice):
    s = sublattice.get_all_atom_planes_by_zone_vector()
    s.plot()
    s.axes_manager.indices = (1,)
    s._plot.signal_plot.figure.savefig(
            os.path.join(my_path, 'sublattice_A_zone1.png'))


def plot_atom_lattice(atom_lattice):
    s = atom_lattice.get_sublattice_atom_list_on_image()
    s.plot()
    s._plot.signal_plot.figure.savefig(
            os.path.join(my_path, 'atom_lattice.png'))


def plot_image_wo_A(image):
    s = hs.signals.Signal2D(image)
    s.plot()
    s._plot.signal_plot.figure.savefig(
            os.path.join(my_path, 'signal_wo_A.png'))


s = dummy_data.get_two_sublattice_signal()
A_positions = am.get_atom_positions(s, separation=15)
sublattice_A = am.Sublattice(A_positions, image=s.data, color='r')
sublattice_A.find_nearest_neighbors()
sublattice_A.refine_atom_positions_using_center_of_mass()
sublattice_A.refine_atom_positions_using_2d_gaussian()
sublattice_A.construct_zone_axes()

save_sub_A_image(sublattice_A)
plot_planes_figure(sublattice_A)

zone_axis_001 = sublattice_A.zones_axis_average_distances[1]
B_positions = sublattice_A.find_missing_atoms_from_zone_vector(zone_axis_001)
image_without_A = remove_atoms_from_image_using_2d_gaussian(
        sublattice_A.image, sublattice_A)

plot_image_wo_A(image_without_A)

sublattice_B = am.Sublattice(B_positions, image_without_A, color='blue')
sublattice_B.construct_zone_axes()
sublattice_B.refine_atom_positions_using_center_of_mass()
sublattice_B.refine_atom_positions_using_2d_gaussian()
atom_lattice = am.Atom_Lattice(
        image=s.data, name='test',
        sublattice_list=[sublattice_A, sublattice_B])

plot_atom_lattice(atom_lattice)
