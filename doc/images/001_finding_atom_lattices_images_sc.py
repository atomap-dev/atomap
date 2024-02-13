import os
import atomap.api as am
import atomap.dummy_data as dummy_data

my_path = os.path.join(os.path.dirname(__file__), 'finding_atom_lattices')
if not os.path.exists(my_path):
    os.makedirs(my_path)


def save_simple_cubic_image(s):
    s.plot()
    s._plot.signal_plot.figure.savefig(
            os.path.join(my_path, 'sc_image.png'))


def save_s_peaks_image(s):
    s_peaks = am.get_feature_separation(s, separation_range=(2, 20))
    s_peaks.plot()
    s_peaks._plot.signal_plot.figure.savefig(
                   os.path.join(my_path, 'peak_finding_1a.png'))
    s_peaks._plot.navigator_plot.figure.savefig(
                   os.path.join(my_path, 'peak_finding_1b.png'))
    s_peaks.axes_manager.indices = (5,)
    s_peaks._plot.signal_plot.figure.savefig(
                   os.path.join(my_path, 'peak_finding_2a.png'))
    s_peaks._plot.navigator_plot.figure.savefig(
                   os.path.join(my_path, 'peak_finding_2b.png'))
    s_peaks.axes_manager.indices = (10,)
    s_peaks._plot.signal_plot.figure.savefig(
                   os.path.join(my_path, 'peak_finding_3a.png'))
    s_peaks._plot.navigator_plot.figure.savefig(
                   os.path.join(my_path, 'peak_finding_3b.png'))


def plot_position_history(sublattice):
    s = sublattice.get_position_history()
    s.plot()
    s._plot.signal_plot.figure.savefig(
                   os.path.join(my_path, 'pos_hist_1a.png'))
    s._plot.navigator_plot.figure.savefig(
                   os.path.join(my_path, 'pos_hist_1b.png'))
    s.axes_manager.indices = (2,)
    s._plot.signal_plot.figure.savefig(
                   os.path.join(my_path, 'pos_hist_2a.png'))
    s._plot.navigator_plot.figure.savefig(
                   os.path.join(my_path, 'pos_hist_2b.png'))
    s._plot.signal_plot.ax.set_xlim(105, 180)
    s._plot.signal_plot.ax.set_ylim(105, 180)
    s._plot.signal_plot.figure.savefig(
                   os.path.join(my_path, 'pos_hist_2_zoom.png'))

def plot_planes_figure(sublattice):
    s = sublattice.get_all_atom_planes_by_zone_vector()
    s.plot()
    s.axes_manager.indices = (1,)
    s._plot.signal_plot.figure.savefig(
                   os.path.join(my_path, 'zone_axes_sig.png'))
    s._plot.navigator_plot.figure.savefig(
                   os.path.join(my_path, 'zone_axes_nav.png'))


s = dummy_data.get_simple_cubic_signal(image_noise=True)

save_simple_cubic_image(s)
save_s_peaks_image(s)

atom_positions = am.get_atom_positions(s, separation=7)
sublattice = am.Sublattice(atom_positions, image=s.data)
sublattice.find_nearest_neighbors()
sublattice.refine_atom_positions_using_center_of_mass()
sublattice.refine_atom_positions_using_2d_gaussian()

plot_position_history(sublattice)

sublattice.construct_zone_axes()
plot_planes_figure(sublattice)
