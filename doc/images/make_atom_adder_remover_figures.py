import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import atomap.api as am
import atomap.animation_plotting_tools as apt

my_path = os.path.join(os.path.dirname(__file__), 'atomadderremovergui')
if not os.path.exists(my_path):
    os.makedirs(my_path)

#####
s = am.dummy_data.get_distorted_cubic_signal()
peaks = am.get_atom_positions(s, 25)
peaks_new = am.add_atoms_with_gui(s, peaks)
ifig = plt.get_fignums()[0]
fig = plt.figure(ifig)
ax = fig.axes[0]

position_list = [
        [-20, 20],
        [51, 215],
        [90, 215],
        [129, 215],
        [159, 220],
        [170, 215],
        [159, 220],
        [211.5, 215],
        [232, 250],
        ]
frames = apt._generate_frames_position_list(position_list, num=10)

fig.dpi = 100
fargs = [fig, ]
apt._draw_cursor(ax, frames[0][0], frames[0][1])
anim = FuncAnimation(fig, apt._update_frame, frames=frames, fargs=fargs,
                     interval=200, repeat=False)
anim.save(os.path.join(my_path, "atoms_add_remove_gui.gif"), writer='pillow')
plt.close(fig)
