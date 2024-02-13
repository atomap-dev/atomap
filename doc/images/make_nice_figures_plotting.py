import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib.patheffects as patheffects
import hyperspy.api as hs

my_path = os.path.join(os.path.dirname(__file__), 'make_nice_figures')
if not os.path.exists(my_path):
    os.makedirs(my_path)

# Load the atomic resolution image
s_adf = hs.load(os.path.join(my_path, 'ADF_image.hspy'))

# Load the structural data
atoms_A = np.load(os.path.join(my_path, 'sublattice_A.npz'))
atoms_B = np.load(os.path.join(my_path, 'sublattice_B.npz'))
dd_map = hs.load(os.path.join(my_path, 'distance_difference_map.hspy'))
dd_line = hs.load(os.path.join(my_path, 'dd_line.hspy'))

# Scaling the data
scale = 0.142
s_adf.axes_manager[0].scale = scale
s_adf.axes_manager[1].scale = scale
# dd_map has twice the amount of pixels, so the scale is half
dd_map.axes_manager[0].scale = scale/2
dd_map.axes_manager[1].scale = scale/2

# Crop images
s_adf = s_adf.isig[40:460, 40:460]
dd_map = dd_map.isig[80:920, 80:920]

# Make a figure with 3 sub-figures, of difference sizes
fig = plt.figure(figsize=(4.3, 2))  # in inches
gs = gridspec.GridSpec(1, 5)
ax_adf = plt.subplot(gs[:2])
ax_al = plt.subplot(gs[2:4])
ax_lp = plt.subplot(gs[4])

# Plot ADF-image
cax_adf = ax_adf.imshow(
        np.rot90(s_adf.data), interpolation='nearest',
        origin='upper', extent=s_adf.axes_manager.signal_extent)

# Make scalebar on ADF-image
fontprops = fm.FontProperties(size=12)
scalebar0 = AnchoredSizeBar(
        ax_adf.transData,
        20, '2 nm', 4,
        pad=0.1,
        color='white',
        frameon=False,
        label_top=True,
        size_vertical=2,
        fontproperties=fontprops)
# The next line is needed due to a bug in matplotlib 2.0
scalebar0.size_bar.get_children()[0].fill = True
ax_adf.add_artist(scalebar0)

# Add markers for atom positions
for idx, x in enumerate(atoms_A['x']):
    y = atoms_A['y'][idx]
    if (240 < x < 350) and (96 < y < 200):
        ax_adf.scatter(y*scale, x*scale, color='r', s=0.5)

for idx, x in enumerate(atoms_B['x']):
    y = atoms_B['y'][idx]
    if (240 < x < 350) and (96 < y < 200):
        ax_adf.scatter(y*scale, x*scale, color='b', s=0.5)

# Plot distance difference map
cax_al = ax_al.imshow(
        np.rot90(dd_map.data),
        interpolation='nearest',
        origin='upper',
        extent=dd_map.axes_manager.signal_extent,
        cmap='viridis')

scalebar1 = AnchoredSizeBar(
        ax_al.transData,
        20, '2 nm', 4,
        pad=0.1,
        color='white',
        frameon=False,
        label_top=True,
        size_vertical=2,
        fontproperties=fontprops)
# The next line is needed due to a bug in matplotlib 2.0
scalebar1.size_bar.get_children()[0].fill = True
ax_al.add_artist(scalebar1)

# Remove ticks for images
for ax in [ax_adf, ax_al]:
    ax.set_xticks([])
    ax.set_yticks([])

# Plot line profile
x_line_profile = dd_line.metadata.line_profile_data.x_list*scale/10
y_line_profile = dd_line.metadata.line_profile_data.y_list*scale
ax_lp.plot(y_line_profile, x_line_profile)
ax_lp.set_xlabel("Distance difference, [Å]", fontsize=7)
ax_lp.set_ylabel("Distance from interface, [nm]", fontsize=7)
ax_lp.tick_params(axis='both', which='major', labelsize=6)
ax_lp.tick_params(axis='both', which='minor', labelsize=6)
ax_lp.yaxis.set_label_position('right')
ax_lp.yaxis.set_ticks_position('right')
ax_lp.set_ylim(-1.5, 4.5)

# Add annotation
path_effects = [
        patheffects.withStroke(linewidth=2, foreground='black',
                               capstyle="round")]
ax_adf.text(
        0.015, 0.90, "a", fontsize=12, color='white',
        path_effects=path_effects,
        transform=ax_adf.transAxes)
ax_al.text(
        0.015, 0.90, "b", fontsize=12, color='white',
        path_effects=path_effects,
        transform=ax_al.transAxes)
ax_lp.text(
        0.05, 0.90, "c", fontsize=12, color='w',
        path_effects=path_effects,
        transform=ax_lp.transAxes)

# Adjust space between subplots and margins
gs.update(left=0.01, wspace=0.05, top=0.95, bottom=0.2, right=0.89)

# Save
fig.savefig(os.path.join(my_path, 'Atom_lattice.png'), dpi=300)
