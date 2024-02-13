import os
from scipy.ndimage import rotate
import atomap.api as am
from atomap.tools import rotate_points_around_signal_centre
import matplotlib.pyplot as plt

my_path = os.path.join(os.path.dirname(__file__), 'makevarioustools')
if not os.path.exists(my_path):
    os.makedirs(my_path)

#####
sublattice = am.dummy_data.get_distorted_cubic_sublattice()
s = sublattice.get_atom_list_on_image()
s_orig = s.deepcopy()
rotation = 30
s.map(rotate, angle=rotation, reshape=False)
x, y = sublattice.x_position, sublattice.y_position
x_rot, y_rot = rotate_points_around_signal_centre(s, x, y, rotation)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
ax0.imshow(s_orig.data, origin='lower',
           extent=s_orig.axes_manager.signal_extent)
ax0.scatter(x, y, s=0.2, color='red')
ax1.imshow(s.data, origin='lower', extent=s.axes_manager.signal_extent)
ax1.scatter(x_rot, y_rot, s=0.2, color='red')

ax1.set_xlim(s.axes_manager[0].low_value, s.axes_manager[0].high_value)
ax1.set_ylim(s.axes_manager[1].low_value, s.axes_manager[1].high_value)

fig.tight_layout()
fig.savefig(os.path.join(my_path, "rotate_image_and_points.png"), dpi=100)

#####
sublattice = am.dummy_data.get_single_atom_sublattice()
sublattice.refine_atom_positions_using_center_of_mass(mask_radius=9)
sublattice.refine_atom_positions_using_2d_gaussian(mask_radius=9)
s = sublattice.get_atom_list_on_image()

s.plot()
filename = os.path.join(my_path, 'single_atom_sublattice.png')
s._plot.signal_plot.figure.savefig(filename)

#####
s = am.dummy_data.get_simple_cubic_signal()
sublattice = am.dummy_data.get_simple_cubic_sublattice()
s_pdf = sublattice.pair_distribution_function(s)
s_pdf.plot()
filename_pdf = os.path.join(my_path, 'pair_distribution_function.png')
s_pdf._plot.signal_plot.figure.savefig(filename_pdf)
