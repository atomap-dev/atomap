import os
import matplotlib.pyplot as plt
import atomap.api as am

my_path = os.path.join(os.path.dirname(__file__), 'plotting_tutorial')
if not os.path.exists(my_path):
    os.makedirs(my_path)


def plot_elli_maps():
    sublattice_A.plot_ellipticity_map(cmap='viridis', vmin=0.95, vmax=1.3)
    plt.gcf().savefig(os.path.join(my_path, 'ellipticity_map_A.png'))

    sublattice_B.plot_ellipticity_map(cmap='viridis', vmin=0.95, vmax=1.3)
    plt.gcf().savefig(os.path.join(my_path, 'ellipticity_map_B.png'))

    sublattice_A.plot_ellipticity_vectors()
    plt.gcf().savefig(os.path.join(my_path, 'ellipticity_vectors.png'))


def plot_monolayer_map():
    s_monolayer = sublattice_B.get_monolayer_distance_map()
    s_monolayer.plot(cmap='viridis')
    s_monolayer.axes_manager.indices = (1,)
    s_monolayer._plot.signal_plot.figure.savefig(
             os.path.join(my_path, 'Sublattice_B_monolayer_distance_a.png'))


def plot_atom_plane_monolayer_map():
    image = atom_lattice.image
    s = sublattice_B.get_all_atom_planes_by_zone_vector(image=image)
    s.plot()
    s.axes_manager.indices = (1,)
    s._plot.signal_plot.figure.savefig(
             os.path.join(my_path, 'Sublattice_B_monolayer_distance_b.png'))


def plot_atom_dd():
    zone = sublattice_B.zones_axis_average_distances[0]
    s_dd = sublattice_B.get_atom_distance_difference_map([zone])
    s_dd.plot(cmap='viridis')
    s_dd._plot.signal_plot.figure.savefig(
             os.path.join(my_path, 'sublatticeB_dd_map_0.png'))

    zone = sublattice_B.zones_axis_average_distances[1]
    s_dd = sublattice_B.get_atom_distance_difference_map(
            [zone], add_zero_value_sublattice=sublattice_A)
    s_dd.plot(cmap='viridis')
    s_dd._plot.signal_plot.figure.savefig(
             os.path.join(my_path, 'sublatticeB_dd_map_1.png'))


def plot_dd_plane():
    image = atom_lattice.image
    s = sublattice_B.get_all_atom_planes_by_zone_vector(image=image)
    s.plot()
    s.axes_manager.indices = (0,)
    s._plot.signal_plot.figure.savefig(
             os.path.join(my_path, 'Angle_map_z1.png'))
    s.axes_manager.indices = (1,)
    s._plot.signal_plot.figure.savefig(
             os.path.join(my_path, 'Angle_map_z2.png'))


def plot_voronoi_integration(atom_lattice):
    integrated = atom_lattice.integrate_column_intensity(
            method='Voronoi')
    s1 = integrated[1]
    s1.plot(cmap='viridis')
    s1._plot.signal_plot.figure.savefig(
            os.path.join(my_path, 'intensity_record_voronoi.png'))


def plot_watershed_integration(atom_lattice):
    integrated = atom_lattice.integrate_column_intensity(
            method='Watershed')
    s1 = integrated[1]
    s1.plot(cmap='viridis')
    s1._plot.signal_plot.figure.savefig(
            os.path.join(my_path, 'intensity_record_watershed.png'))


def plot_angle_figs():
    z1 = sublattice_B.zones_axis_average_distances[0]
    z2 = sublattice_B.zones_axis_average_distances[1]
    x, y, a = sublattice_B.get_atom_angles_from_zone_vector(
            z1, z2, degrees=True)
    s_angle = sublattice_B.get_property_map(x, y, a)
    s_angle.plot(cmap='magma')
    s_angle._plot.signal_plot.figure.savefig(
             os.path.join(my_path, 'Angle_map.png'))


def plot_al_zoom():
    s = atom_lattice.get_sublattice_atom_list_on_image()
    s.plot()
    s._plot.signal_plot.ax.set_xlim(364, 437)
    s._plot.signal_plot.ax.set_ylim(184, 259)
    s._plot.signal_plot.figure.savefig(
             os.path.join(my_path, 'Angle_map_zoom.png'))


def elli_line():
    zone = sublattice_A.zones_axis_average_distances[1]
    plane = sublattice_A.atom_planes_by_zone_vector[zone][23]
    s_elli_line = sublattice_A.get_ellipticity_line_profile(plane)
    s_elli_line.plot()
    s_elli_line._plot.signal_plot.figure.savefig(
             os.path.join(my_path, 'line_ellip.png'))


def elli_line_errorbar():
    zone = sublattice_A.zones_axis_average_distances[1]
    plane = sublattice_A.atom_planes_by_zone_vector[zone][23]
    s_elli_line = sublattice_A.get_ellipticity_line_profile(plane)
    x = s_elli_line.metadata.line_profile_data.x_list
    y = s_elli_line.metadata.line_profile_data.y_list
    std = s_elli_line.metadata.line_profile_data.std_list
    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=std, ecolor='red')
    fig.savefig(os.path.join(my_path, 'line_profile_errorbar.png'))


def plot_line_plane():
    s = sublattice_A.get_all_atom_planes_by_zone_vector()
    s.plot()
    s.axes_manager.indices = (1,)
    s._plot.signal_plot.figure.savefig(
             os.path.join(my_path, 'line_ellip_plane.png'))


def monolayer_line():
    zone = sublattice_B.zones_axis_average_distances[1]
    plane = sublattice_B.atom_planes_by_zone_vector[zone][0]
    s_line = sublattice_B.get_monolayer_distance_line_profile(zone, plane)
    s_line.plot()
    s_line._plot.signal_plot.figure.savefig(
             os.path.join(my_path, 'line_monolayer.png'))


def dd_line():
    zone = sublattice_B.zones_axis_average_distances[1]
    plane = sublattice_B.atom_planes_by_zone_vector[zone][-1]
    s_line = sublattice_B.get_atom_distance_difference_line_profile(
            zone, plane)
    s_line.plot()
    s_line._plot.signal_plot.figure.savefig(
             os.path.join(my_path, 'line_dd.png'))


atom_lattice = am.load_atom_lattice_from_hdf5(
        os.path.join(my_path, 'fantasite.hdf5'))
sublattice_A = atom_lattice.sublattice_list[0]
sublattice_B = atom_lattice.sublattice_list[1]

plot_elli_maps()
plot_monolayer_map()
plot_atom_plane_monolayer_map()
plot_atom_dd()
plot_dd_plane()
plot_voronoi_integration(atom_lattice)
plot_watershed_integration(atom_lattice)
plot_angle_figs()
plot_al_zoom()
elli_line()
elli_line_errorbar()
plot_line_plane()
monolayer_line()
dd_line()
