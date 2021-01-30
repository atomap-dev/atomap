import os
import atomap.api as am
import matplotlib.pyplot as plt

my_path = os.path.join(os.path.dirname(__file__), 'quant')
if not os.path.exists(my_path):
    os.makedirs(my_path)

# Making and fitting data
s = am.dummy_data.get_atom_counting_signal()
atom_positions = am.get_atom_positions(s, 8, threshold_rel=0.1)
sublattice = am.Sublattice(atom_positions, s)
sublattice.construct_zone_axes()
sublattice.refine_atom_positions_using_2d_gaussian()

# Plotting
models = am.quant.get_statistical_quant_criteria([sublattice], 10)
plt.savefig(os.path.join(my_path, 'criteria_plot.png'), overwrite=True)

atom_lattice = am.quant.statistical_quant(sublattice, models[3], 4, 'C', 2.4)
plt.savefig(os.path.join(my_path, 'quant_output1a.png'), overwrite=True)

s_al = atom_lattice.get_sublattice_atom_list_on_image()
s_al.plot()
s_al._plot.signal_plot.figure.savefig(
        os.path.join(my_path, 'quant_output1b.png'), dpi=150)
