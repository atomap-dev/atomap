import os
import numpy as np
from hyperspy.signals import Signal2D
from atomap.dummy_data import get_fantasite_atom_lattice

my_path = os.path.join(os.path.dirname(__file__), 'make_nice_figures')
if not os.path.exists(my_path):
    os.makedirs(my_path)

# First we will analyse and save the structural data of interest
# Here, we use the fantasite atom_lattice dummy data
atom_lattice = get_fantasite_atom_lattice()

# Saving atom positions and ellipticity
sublattice_A = atom_lattice.sublattice_list[0]
np.savez(
        os.path.join(my_path, 'sublattice_A.npz'), x=sublattice_A.x_position,
        y=sublattice_A.y_position, e=sublattice_A.ellipticity)
sublattice_B = atom_lattice.sublattice_list[1]
np.savez(
        os.path.join(my_path, 'sublattice_B.npz'), x=sublattice_B.x_position,
        y=sublattice_B.y_position, e=sublattice_B.ellipticity)

# Saving distance difference map
sublattice_A.construct_zone_axes()
zone = sublattice_A.zones_axis_average_distances[0]
s_dd = sublattice_A.get_atom_distance_difference_map([zone])
s_dd.save(os.path.join(my_path, 'distance_difference_map.hspy'),
          overwrite=True)

# Saving the synthetic ADF-image.
im = atom_lattice.image
s_adf = Signal2D(im)
s_adf.save(os.path.join(my_path, 'ADF_image.hspy'), overwrite=True)

# Saving the line profile
z1 = sublattice_A.zones_axis_average_distances[0]
z2 = sublattice_A.zones_axis_average_distances[1]
plane = sublattice_A.atom_planes_by_zone_vector[z2][23]
s_dd_line = sublattice_A.get_atom_distance_difference_line_profile(z1, plane)
s_dd_line.save(os.path.join(my_path, 'dd_line.hspy'), overwrite=True)
