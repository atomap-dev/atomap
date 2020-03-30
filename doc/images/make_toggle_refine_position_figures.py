import os
import matplotlib
matplotlib.use('Qt5Agg') # noqa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import atomap.api as am
import atomap.animation_plotting_tools as apt

my_path = os.path.join(os.path.dirname(__file__), 'togglerefineposition')
if not os.path.exists(my_path):
    os.makedirs(my_path)

#####
sublattice = am.dummy_data.get_distorted_cubic_sublattice()
sublattice.toggle_atom_refine_position_with_gui()
ifig = plt.get_fignums()[0]
fig = plt.figure(ifig)
ax = fig.axes[0]

position_list = [
        [-20, 20],
        [70, 90],
        [150, 70],
        [90, 135],
        [70, 90],
        [50, 30],
        [64, -20],
        ]
frames = apt._generate_frames_position_list(position_list, num=10)

fig.dpi = 100
fargs = [fig, ]
apt._draw_cursor(ax, frames[0][0], frames[0][1])
anim = FuncAnimation(fig, apt._update_frame, frames=frames, fargs=fargs,
                     interval=200, repeat=False)
anim.save(os.path.join(
    my_path, "toggle_refine_position.gif"), writer='pillow')
plt.close(fig)
