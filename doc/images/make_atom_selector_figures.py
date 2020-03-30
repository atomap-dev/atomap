import os
import matplotlib
matplotlib.use('Qt5Agg') # noqa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import atomap.api as am
import atomap.animation_plotting_tools as apt
import atomap.gui_classes as agc

my_path = os.path.join(os.path.dirname(__file__), 'atomselectorgui')
if not os.path.exists(my_path):
    os.makedirs(my_path)

#####
s = am.dummy_data.get_precipitate_signal()
peaks = am.get_atom_positions(s, 8)

atom_selector = agc.GetAtomSelection(s, peaks)
poly = atom_selector.poly
fig = atom_selector.fig
ax = fig.axes[0]

position_list = [[250, 100], [100, 250], [250, 400], [400, 250],
                 [250, 100], [-10, 80]]
frames = apt._generate_frames_position_list(position_list, num=10)

fig.dpi = 100
fargs = [fig, poly]
apt._draw_cursor(ax, frames[0][0], frames[0][1])
anim = FuncAnimation(fig, apt._update_frame_poly, frames=frames, fargs=fargs,
                     interval=400, repeat=False)
anim.save(os.path.join(my_path, "atom_selector_gui.gif"), writer='pillow')
plt.close(fig)

#####
s = am.dummy_data.get_precipitate_signal()
peaks = am.get_atom_positions(s, 8)

atom_selector = agc.GetAtomSelection(s, peaks, invert_selection=True)
poly = atom_selector.poly
fig = atom_selector.fig
ax = fig.axes[0]

position_list = [[250, 100], [100, 250], [250, 400], [400, 250],
                 [250, 100], [-10, 80]]
frames = apt._generate_frames_position_list(position_list, num=10)

fig.dpi = 100
fargs = [fig, poly]
apt._draw_cursor(ax, frames[0][0], frames[0][1])
anim = FuncAnimation(fig, apt._update_frame_poly, frames=frames, fargs=fargs,
                     interval=400, repeat=False)
anim.save(os.path.join(my_path, "atom_selector_invert_selection_gui.gif"), writer='pillow')
plt.close(fig)
