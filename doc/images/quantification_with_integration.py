import os
import shutil
import requests

import matplotlib.pyplot as plt
import atomap.api as am
import hyperspy.api as hs

my_path = os.path.join(os.path.dirname(__file__), 'quantification')
if not os.path.exists(my_path):
    os.makedirs(my_path)

# image 1
s = hs.load(os.path.join(my_path, 'simulated_nanoparticle.tif'))
points_x, points_y = am.get_atom_positions(s, separation=4).T
i_points, i_record, p_record = am.integrate(s, points_x, points_y)
i_record.plot(cmap='viridis')
plt.gcf().savefig(os.path.join(my_path, 'voronoi_nanoparticle1.png'))

# image 2
from atomap.tools import remove_integrated_edge_cells
i_points, i_record, p_record = remove_integrated_edge_cells(i_points, i_record, p_record, edge_pixels=30)
i_record.plot(cmap='viridis')
plt.gcf().savefig(os.path.join(my_path, 'voronoi_nanoparticle_remove_edge.png'))

# image 3
points_x, points_y = am.get_atom_positions(s, separation=4).T
i_points, i_record, p_record = am.integrate(s, points_x, points_y, max_radius=5)
i_record.plot(cmap='viridis')
plt.gcf().savefig(os.path.join(my_path, 'voronoi_nanoparticle_max_radius.png'))
