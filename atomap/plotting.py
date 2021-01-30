import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import hsv_to_rgb
import copy
from hyperspy.drawing._markers.line_segment import LineSegment
from hyperspy.drawing._markers.point import Point
from hyperspy.drawing._markers.text import Text

import atomap.tools as to


def plot_vector_field(x_pos_list, y_pos_list, x_rot_list, y_rot_list, save=True):
    x_shape = min(x_pos_list), max(x_pos_list)
    y_shape = min(y_pos_list), max(y_pos_list)
    ar = (x_shape[1] - x_shape[0]) / (y_shape[1] - y_shape[0])
    fig, ax = plt.subplots(figsize=(6 * ar, 6))
    ax.quiver(
        x_pos_list,
        y_pos_list,
        x_rot_list,
        y_rot_list,
        headwidth=0.0,
        headlength=0.0,
        headaxislength=0.0,
        scale=20.0,
        pivot="middle",
    )
    ax.set_xlim(min(x_pos_list), max(x_pos_list))
    ax.set_ylim(min(y_pos_list), max(y_pos_list))
    ax.set_aspect("equal")
    if save:
        fig.savefig("vector_field.png", dpi=200)


def plot_zone_vector_and_atom_distance_map(
    image_data,
    distance_data,
    atom_planes=None,
    distance_data_scale=1,
    atom_list=None,
    extra_marker_list=None,
    clim=None,
    atom_plane_marker=None,
    plot_title="",
    vector_to_plot=None,
    figsize=(10, 20),
    figname="map_data.jpg",
):
    """
    Parameters
    ----------
    atom_list : list of Atom_Position instances
    extra_marker_list : two arrays of x and y [[x_values], [y_values]]
    """
    if image_data is None:
        raise ValueError("Image data is None, no data to plot")

    fig = Figure(figsize=figsize)
    FigureCanvas(fig)

    gs = GridSpec(95, 95)

    image_ax = fig.add_subplot(gs[0:45, :])
    distance_ax = fig.add_subplot(gs[45:90, :])
    colorbar_ax = fig.add_subplot(gs[90:, :])

    image_clim = to._get_clim_from_data(image_data, sigma=2)
    image_cax = image_ax.imshow(image_data)
    image_cax.set_clim(image_clim[0], image_clim[1])
    if atom_planes:
        for atom_plane_index, atom_plane in enumerate(atom_planes):
            x_pos = atom_plane.get_x_position_list()
            y_pos = atom_plane.get_y_position_list()
            image_ax.plot(x_pos, y_pos, lw=3, color="blue")
            image_ax.text(
                atom_plane.start_atom.pixel_x,
                atom_plane.start_atom.pixel_y,
                str(atom_plane_index),
                color="red",
            )
    image_ax.set_ylim(0, image_data.shape[0])
    image_ax.set_xlim(0, image_data.shape[1])
    image_ax.set_title(plot_title)

    if atom_plane_marker:
        atom_plane_x = atom_plane_marker.get_x_position_list()
        atom_plane_y = atom_plane_marker.get_y_position_list()
        image_ax.plot(atom_plane_x, atom_plane_y, color="red", lw=2)

    _make_subplot_map_from_regular_grid(
        distance_ax,
        distance_data,
        distance_data_scale=distance_data_scale,
        clim=clim,
        atom_list=atom_list,
        atom_plane_marker=atom_plane_marker,
        extra_marker_list=extra_marker_list,
        vector_to_plot=vector_to_plot,
    )
    distance_cax = distance_ax.images[0]

    fig.tight_layout()
    fig.colorbar(distance_cax, cax=colorbar_ax, orientation="horizontal")
    fig.savefig(figname)
    plt.close(fig)


def plot_complex_image_map_line_profile_using_interface_plane(
    image_data,
    amplitude_data,
    phase_data,
    line_profile_amplitude_data,
    line_profile_phase_data,
    interface_plane,
    atom_plane_list=None,
    data_scale=1,
    atom_list=None,
    extra_marker_list=None,
    amplitude_image_lim=None,
    phase_image_lim=None,
    plot_title="",
    add_color_wheel=False,
    color_bar_markers=None,
    vector_to_plot=None,
    rotate_atom_plane_list_90_degrees=False,
    prune_outer_values=False,
    figname="map_data.jpg",
):
    """
    Parameters
    ----------
    atom_list : list of Atom_Position instances
    extra_marker_list : two arrays of x and y [[x_values], [y_values]]
    """
    number_of_line_profiles = 2

    figsize = (10, 18 + 2 * number_of_line_profiles)

    fig = Figure(figsize=figsize)
    FigureCanvas(fig)

    gs = GridSpec(100 + 10 * number_of_line_profiles, 95)

    image_ax = fig.add_subplot(gs[0:45, :])
    distance_ax = fig.add_subplot(gs[45:90, :])
    colorbar_ax = fig.add_subplot(gs[90:100, :])

    line_profile_ax_list = []
    for i in range(number_of_line_profiles):
        gs_y_start = 100 + 10 * i
        line_profile_ax = fig.add_subplot(gs[gs_y_start : gs_y_start + 10, :])
        line_profile_ax_list.append(line_profile_ax)

    image_y_lim = (0, image_data.shape[0] * data_scale)
    image_x_lim = (0, image_data.shape[1] * data_scale)

    image_clim = to._get_clim_from_data(image_data, sigma=2)
    image_cax = image_ax.imshow(
        image_data,
        origin="lower",
        extent=[image_x_lim[0], image_x_lim[1], image_y_lim[0], image_y_lim[1]],
    )

    image_cax.set_clim(image_clim[0], image_clim[1])
    image_ax.set_xlim(image_x_lim[0], image_x_lim[1])
    image_ax.set_ylim(image_y_lim[0], image_y_lim[1])
    image_ax.set_title(plot_title)

    if atom_plane_list is not None:
        for atom_plane in atom_plane_list:
            if rotate_atom_plane_list_90_degrees:
                atom_plane_x = np.array(atom_plane.get_x_position_list())
                atom_plane_y = np.array(atom_plane.get_y_position_list())
                start_x = atom_plane_x[0]
                start_y = atom_plane_y[0]
                delta_y = atom_plane_x[-1] - atom_plane_x[0]
                delta_x = -(atom_plane_y[-1] - atom_plane_y[0])
                atom_plane_x = np.array([start_x, start_x + delta_x])
                atom_plane_y = np.array([start_y, start_y + delta_y])
            else:
                atom_plane_x = np.array(atom_plane.get_x_position_list())
                atom_plane_y = np.array(atom_plane.get_y_position_list())
            image_ax.plot(
                atom_plane_x * data_scale, atom_plane_y * data_scale, color="red", lw=2
            )

    atom_plane_x = np.array(interface_plane.get_x_position_list())
    atom_plane_y = np.array(interface_plane.get_y_position_list())
    image_ax.plot(
        atom_plane_x * data_scale, atom_plane_y * data_scale, color="blue", lw=2
    )

    _make_subplot_map_from_complex_regular_grid(
        distance_ax,
        amplitude_data,
        phase_data,
        atom_list=atom_list,
        amplitude_image_lim=amplitude_image_lim,
        phase_image_lim=phase_image_lim,
        atom_plane_marker=interface_plane,
        extra_marker_list=extra_marker_list,
        vector_to_plot=vector_to_plot,
    )
    distance_ax.plot(
        atom_plane_x * data_scale, atom_plane_y * data_scale, color="red", lw=2
    )

    line_profile_data_list = [line_profile_amplitude_data, line_profile_phase_data]

    for line_profile_ax, line_profile_data in zip(
        line_profile_ax_list, line_profile_data_list
    ):
        _make_subplot_line_profile(
            line_profile_ax,
            line_profile_data[:, 0],
            line_profile_data[:, 1],
            prune_outer_values=prune_outer_values,
            scale_x=data_scale,
        )

    amplitude_delta = 0.01 * (amplitude_image_lim[1] - amplitude_image_lim[0])
    phase_delta = 0.01 * (phase_image_lim[1] - phase_image_lim[0])
    colorbar_mgrid = np.mgrid[
        amplitude_image_lim[0] : amplitude_image_lim[1] : amplitude_delta,
        phase_image_lim[0] : phase_image_lim[1] : phase_delta,
    ]
    colorbar_rgb = get_rgb_array(colorbar_mgrid[1], colorbar_mgrid[0])
    colorbar_ax.imshow(
        colorbar_rgb,
        origin="lower",
        extent=[
            phase_image_lim[0],
            phase_image_lim[1],
            amplitude_image_lim[0],
            amplitude_image_lim[1],
        ],
    )

    colorbar_ax.set_xlabel("Phase", size=6)

    if color_bar_markers is not None:
        for color_bar_marker in color_bar_markers:
            colorbar_ax.axvline(color_bar_marker[0], color="white")
            colorbar_ax.text(
                color_bar_marker[0],
                amplitude_image_lim[0] * 0.97,
                color_bar_marker[1],
                transform=colorbar_ax.transData,
                va="top",
                ha="center",
                fontsize=8,
            )

    if add_color_wheel:
        ax_magnetic_color_wheel_gs = GridSpecFromSubplotSpec(
            40, 40, subplot_spec=gs[45:90, :]
        )[30:39, 3:12]
        ax_magnetic_color_wheel = fig.add_subplot(ax_magnetic_color_wheel_gs)
        ax_magnetic_color_wheel.set_axis_off()

        make_color_wheel(ax_magnetic_color_wheel)

    fig.tight_layout()
    fig.savefig(figname)
    plt.close(fig)


def normalize_array(np_array, max_number=1.0):
    np_array = copy.deepcopy(np_array)
    np_array -= np_array.min()
    np_array /= np_array.max()
    return np_array * max_number


def get_rgb_array(angle, magnitude, rotation=0, angle_lim=None, magnitude_lim=None):
    if not (rotation == 0):
        angle = ((angle + np.radians(rotation) + np.pi) % (2 * np.pi)) - np.pi
    if angle_lim is not None:
        np.clip(angle, angle_lim[0], angle_lim[1], out=angle)
        angle = angle / angle.max()
    else:
        angle = normalize_array(angle)
    if magnitude_lim is not None:
        np.clip(magnitude, magnitude_lim[0], magnitude_lim[1], out=magnitude)
    magnitude = normalize_array(magnitude)
    S = np.ones_like(angle)
    HSV = np.dstack((angle, S, magnitude))
    RGB = hsv_to_rgb(HSV)
    return RGB


def make_color_wheel(ax, rotation=0):
    x, y = np.mgrid[-2.0:2.0:500j, -2.0:2.0:500j]
    r = (x ** 2 + y ** 2) ** 0.5
    t = np.arctan2(x, y)
    del x, y
    if not (rotation == 0):
        t += np.radians(rotation)
        t = (t + np.pi) % (2 * np.pi) - np.pi

    r_masked = np.ma.masked_where((2.0 < r) | (r < 1.0), r)
    r_masked -= 1.0

    mask = r_masked.mask
    r_masked.data[r_masked.mask] = r_masked.mean()
    rgb_array = get_rgb_array(t, r_masked.data)
    rgb_array = np.dstack((rgb_array, np.invert(mask)))

    ax.imshow(rgb_array, interpolation="quadric", origin="lower")
    ax.set_axis_off()


def plot_image_map_line_profile_using_interface_plane(
    image_data,
    heatmap_data_list,
    line_profile_data_list,
    interface_plane,
    atom_plane_list=None,
    data_scale=1,
    data_scale_z=1,
    atom_list=None,
    extra_marker_list=None,
    clim=None,
    plot_title="",
    vector_to_plot=None,
    rotate_atom_plane_list_90_degrees=False,
    prune_outer_values=False,
    figname="map_data.jpg",
):
    """
    Parameters
    ----------
    atom_list : list of Atom_Position instances
    extra_marker_list : two arrays of x and y [[x_values], [y_values]]
    """
    number_of_line_profiles = len(line_profile_data_list)

    figsize = (10, 18 + 2 * number_of_line_profiles)

    fig = Figure(figsize=figsize)
    FigureCanvas(fig)

    gs = GridSpec(95 + 10 * number_of_line_profiles, 95)

    image_ax = fig.add_subplot(gs[0:45, :])
    distance_ax = fig.add_subplot(gs[45:90, :])
    colorbar_ax = fig.add_subplot(gs[90:95, :])
    line_profile_ax_list = []
    for i in range(number_of_line_profiles):
        gs_y_start = 95 + 10 * i
        line_profile_ax = fig.add_subplot(gs[gs_y_start : gs_y_start + 10, :])
        line_profile_ax_list.append(line_profile_ax)

    image_y_lim = (0, image_data.shape[0] * data_scale)
    image_x_lim = (0, image_data.shape[1] * data_scale)

    image_clim = to._get_clim_from_data(image_data, sigma=2)
    image_cax = image_ax.imshow(
        image_data,
        origin="lower",
        extent=[image_x_lim[0], image_x_lim[1], image_y_lim[0], image_y_lim[1]],
    )

    image_cax.set_clim(image_clim[0], image_clim[1])
    image_ax.set_xlim(image_x_lim[0], image_x_lim[1])
    image_ax.set_ylim(image_y_lim[0], image_y_lim[1])
    image_ax.set_title(plot_title)

    if atom_plane_list is not None:
        for atom_plane in atom_plane_list:
            if rotate_atom_plane_list_90_degrees:
                atom_plane_x = np.array(atom_plane.get_x_position_list())
                atom_plane_y = np.array(atom_plane.get_y_position_list())
                start_x = atom_plane_x[0]
                start_y = atom_plane_y[0]
                delta_y = atom_plane_x[-1] - atom_plane_x[0]
                delta_x = -(atom_plane_y[-1] - atom_plane_y[0])
                atom_plane_x = np.array([start_x, start_x + delta_x])
                atom_plane_y = np.array([start_y, start_y + delta_y])
            else:
                atom_plane_x = np.array(atom_plane.get_x_position_list())
                atom_plane_y = np.array(atom_plane.get_y_position_list())
            image_ax.plot(
                atom_plane_x * data_scale, atom_plane_y * data_scale, color="red", lw=2
            )

    atom_plane_x = np.array(interface_plane.get_x_position_list())
    atom_plane_y = np.array(interface_plane.get_y_position_list())
    image_ax.plot(
        atom_plane_x * data_scale, atom_plane_y * data_scale, color="blue", lw=2
    )

    _make_subplot_map_from_regular_grid(
        distance_ax,
        heatmap_data_list,
        distance_data_scale=data_scale_z,
        clim=clim,
        atom_list=atom_list,
        atom_plane_marker=interface_plane,
        extra_marker_list=extra_marker_list,
        vector_to_plot=vector_to_plot,
    )
    distance_cax = distance_ax.images[0]
    distance_ax.plot(
        atom_plane_x * data_scale, atom_plane_y * data_scale, color="red", lw=2
    )

    for line_profile_ax, line_profile_data in zip(
        line_profile_ax_list, line_profile_data_list
    ):
        _make_subplot_line_profile(
            line_profile_ax,
            line_profile_data[:, 0],
            line_profile_data[:, 1],
            prune_outer_values=prune_outer_values,
            scale_x=data_scale,
            scale_z=data_scale_z,
        )

    fig.tight_layout()
    fig.colorbar(distance_cax, cax=colorbar_ax, orientation="horizontal")
    fig.savefig(figname)


def _make_figure_scatter_point_on_image(
    image, x, y, z, cmap=None, vmin=None, vmax=None
):
    """Make a figure with values overlayed as points on an image.

    Parameters
    ----------
    image : array-like
    x, y, z : list
        Need to have the same shape.
    cmap : string
        Matplotlib colormap name, default 'viridis'
    vmin, vmax : scalars
        Min and max values for the scatter points

    Returns
    -------
    fig : matplotlib figure

    Example
    -------
    >>> import atomap.plotting as apl
    >>> image = np.random.random((100, 100))
    >>> x, y = np.arange(50), np.arange(50)
    >>> z = np.random.randint(1, 9, size=50)
    >>> fig = apl._make_figure_scatter_point_on_image(image, x, y, z)

    """
    if cmap is None:
        cmap = "viridis"
    if vmin is None:
        vmin = z.min()
    if vmax is None:
        vmax = z.max()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(image)
    ax.scatter(x, y, c=z, cmap=cmap, vmin=vmin, vmax=vmax)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax)
    fig.tight_layout()
    return fig


def _make_subplot_line_profile(
    ax,
    x_list,
    y_list,
    scale_x=1.0,
    scale_z=1.0,
    x_lim=None,
    prune_outer_values=False,
    y_lim=None,
):
    x_data_list = x_list * scale_x
    y_data_list = y_list * scale_z
    if prune_outer_values is not False:
        x_data_list = x_data_list[prune_outer_values:-prune_outer_values]
        y_data_list = y_data_list[prune_outer_values:-prune_outer_values]
    ax.plot(x_data_list, y_data_list)
    ax.grid()
    if x_lim is None:
        ax.set_xlim(x_data_list.min(), x_data_list.max())
    else:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim is None:
        ax.set_ylim(y_data_list.min(), y_data_list.max())
    else:
        ax.set_ylim(y_lim[0], y_lim[1])
    ax.axvline(0, color="red")


def _make_subplot_map_from_regular_grid(
    ax,
    data,
    atom_list=None,
    distance_data_scale=1.0,
    clim=None,
    atom_plane_marker=None,
    extra_marker_list=None,
    plot_title="",
    vector_to_plot=None,
):
    """ Data in the form [(x, y, z)]"""
    x_lim = (data[0][0][0], data[0][-1][0])
    y_lim = (data[1][0][0], data[1][0][-1])
    cax = ax.imshow(
        data[2].T * distance_data_scale,
        extent=[
            x_lim[0] * distance_data_scale,
            x_lim[1] * distance_data_scale,
            y_lim[0] * distance_data_scale,
            y_lim[1] * distance_data_scale,
        ],
        cmap="viridis",
        origin="lower",
    )
    if atom_plane_marker:
        atom_plane_x = np.array(atom_plane_marker.get_x_position_list())
        atom_plane_y = np.array(atom_plane_marker.get_y_position_list())
        ax.plot(
            atom_plane_x * distance_data_scale,
            atom_plane_y * distance_data_scale,
            color="red",
            lw=2,
        )
    if atom_list:
        x = []
        y = []
        for atom in atom_list:
            x.append(atom.pixel_x * distance_data_scale)
            y.append(atom.pixel_y * distance_data_scale)
        ax.scatter(x, y)
    if extra_marker_list:
        ax.scatter(extra_marker_list[0], extra_marker_list[1], color="red")
    if clim:
        cax.set_clim(clim[0], clim[1])

    if vector_to_plot:
        x0 = x_lim[0] + (x_lim[1] - x_lim[0]) / 2 * 0.15
        y0 = y_lim[0] + (y_lim[1] - y_lim[0]) / 2 * 0.15
        ax.arrow(
            x0 * distance_data_scale,
            y0 * distance_data_scale,
            vector_to_plot[0] * distance_data_scale,
            vector_to_plot[1] * distance_data_scale,
            width=0.20,
        )
    ax.set_xlim(
        data[0][0][0] * distance_data_scale, data[0][-1][0] * distance_data_scale
    )
    ax.set_ylim(
        data[1][0][0] * distance_data_scale, data[1][0][-1] * distance_data_scale
    )


def _make_subplot_map_from_complex_regular_grid(
    ax,
    amplitude_data,
    phase_data,
    atom_list=None,
    distance_data_scale=1.0,
    atom_plane_marker=None,
    amplitude_image_lim=None,
    phase_image_lim=None,
    extra_marker_list=None,
    plot_title="",
    vector_to_plot=None,
):
    """ amplitude_data and phase_data in the form [(x, y , z)]"""
    x_lim = (amplitude_data[0][0][0], amplitude_data[0][-1][0])
    y_lim = (amplitude_data[1][0][0], amplitude_data[1][0][-1])
    rgb_array = get_rgb_array(
        phase_data[2],
        amplitude_data[2],
        rotation=0,
        angle_lim=phase_image_lim,
        magnitude_lim=amplitude_image_lim,
    )
    ax.imshow(
        np.fliplr(np.rot90(rgb_array, -1)),
        extent=[
            x_lim[0] * distance_data_scale,
            x_lim[1] * distance_data_scale,
            y_lim[0] * distance_data_scale,
            y_lim[1] * distance_data_scale,
        ],
        origin="lower",
    )
    if atom_plane_marker:
        atom_plane_x = np.array(atom_plane_marker.get_x_position_list())
        atom_plane_y = np.array(atom_plane_marker.get_y_position_list())
        ax.plot(
            atom_plane_x * distance_data_scale,
            atom_plane_y * distance_data_scale,
            color="red",
            lw=2,
        )
    if atom_list:
        x = []
        y = []
        for atom in atom_list:
            x.append(atom.pixel_x * distance_data_scale)
            y.append(atom.pixel_y * distance_data_scale)
        ax.scatter(x, y)
    if extra_marker_list:
        ax.scatter(extra_marker_list[0], extra_marker_list[1], color="red")
    if vector_to_plot:
        x0 = x_lim[0] + (x_lim[1] - x_lim[0]) / 2 * 0.15
        y0 = y_lim[0] + (y_lim[1] - y_lim[0]) / 2 * 0.15
        ax.arrow(
            x0 * distance_data_scale,
            y0 * distance_data_scale,
            vector_to_plot[0] * distance_data_scale,
            vector_to_plot[1] * distance_data_scale,
            width=0.20,
        )
    ax.set_xlim(
        amplitude_data[0][0][0] * distance_data_scale,
        amplitude_data[0][-1][0] * distance_data_scale,
    )
    ax.set_ylim(
        amplitude_data[1][0][0] * distance_data_scale,
        amplitude_data[1][0][-1] * distance_data_scale,
    )


def _make_line_profile_subplot_from_three_parameter_data(
    ax, data_list, interface_plane, scale_x=1.0, scale_y=1.0, invert_line_profiles=False
):

    projected_positions = to.project_position_property(data_list, interface_plane)
    layer_list = to.sort_projected_positions_into_layers(projected_positions)
    line_profile_data = to.combine_projected_positions_layers(layer_list)

    line_profile_data = np.array(line_profile_data)

    position = line_profile_data[:, 0]
    data = line_profile_data[:, 1]
    if invert_line_profiles:
        position = position * -1

    _make_subplot_line_profile(ax, position, data, scale_x=scale_x, scale_y=scale_y)


# Parameter list in the form of [position, data]
def plot_line_profiles_from_parameter_input(
    parameter_list,
    parameter_name_list=None,
    invert_line_profiles=False,
    extra_line_marker_list=None,
    x_lim=False,
    figname="line_profile_list.jpg",
):
    figsize = (15, len(parameter_list) * 3)
    fig = Figure(figsize=figsize)
    FigureCanvas(fig)

    gs = GridSpec(10 * len(parameter_list), 10)
    line_profile_gs_size = 10
    for index, parameter in enumerate(parameter_list):
        ax = fig.add_subplot(
            gs[index * line_profile_gs_size : (index + 1) * line_profile_gs_size, :]
        )
        position = parameter[0]
        if invert_line_profiles:
            position = position * -1
        _make_subplot_line_profile(ax, position, parameter[1], scale_x=1.0, scale_y=1.0)
        if parameter_name_list is not None:
            ax.set_ylabel(parameter_name_list[index])

    if x_lim is False:
        x_min = 100000000000
        x_max = -10000000000
        for ax in fig.axes:
            ax_xlim = ax.get_xlim()
            if ax_xlim[0] < x_min:
                x_min = ax_xlim[0]
            if ax_xlim[1] > x_max:
                x_max = ax_xlim[1]
        for ax in fig.axes:
            ax.set_xlim(x_min, x_max)
    else:
        for ax in fig.axes:
            ax.set_xlim(x_lim[0], x_lim[1])

    if extra_line_marker_list is not None:
        for extra_line_marker in extra_line_marker_list:
            for ax in fig.axes:
                ax.axvline(extra_line_marker, color="red")
    fig.tight_layout()
    fig.savefig(figname, dpi=100)


def plot_feature_density(
    separation_value_list, peakN_list, figname="feature_density.png"
):
    fig, ax = plt.subplots()
    ax.plot(separation_value_list, peakN_list)
    ax.set_xlim(separation_value_list[0], separation_value_list[-1])
    ax.set_ylim(peakN_list[0], peakN_list[-1])
    ax.set_xlabel("Feature separation, (pixels)")
    ax.set_ylabel("Feature density, (#)")
    fig.tight_layout()
    fig.savefig("feature_density.png", dpi=200)


def _make_atom_planes_marker_list(
    atom_plane_list, scale=1.0, add_numbers=True, color="red"
):
    marker_list = []
    for i, atom_plane in enumerate(atom_plane_list):
        atom_plane_markers = _make_single_atom_plane_marker_list(
            atom_plane, scale=scale, color="red"
        )
        marker_list.extend(atom_plane_markers)
        if add_numbers:
            marker = Text(
                x=atom_plane.start_atom.pixel_x * scale,
                y=atom_plane.start_atom.pixel_y * scale,
                text=str(i),
                color=color,
                va="top",
                ha="right",
            )
            marker_list.append(marker)
    return marker_list


def _make_atom_position_marker_list(
    atom_position_list, scale=1.0, markersize=20, add_numbers=True, color="red"
):
    marker_list = []
    for i, atom_position in enumerate(atom_position_list):
        x = atom_position.pixel_x * scale
        y = atom_position.pixel_y * scale
        marker = Point(x=x, y=y, color=color, size=markersize)
        marker_list.append(marker)
        if add_numbers:
            marker = Text(x=x, y=y, text=str(i), color=color, va="top", ha="right")
            marker_list.append(marker)
    return marker_list


def _make_multidim_atom_plane_marker_list(
    atom_plane_zone_list, scale=1.0, color="red", add_numbers=True
):
    marker_list = []
    for i, atom_plane_list in enumerate(atom_plane_zone_list):
        for index_atom_plane, atom_plane in enumerate(atom_plane_list):
            for j in range(len(atom_plane.atom_list[1:])):
                x1 = [-1000] * len(atom_plane_zone_list)
                y1 = [-1000] * len(atom_plane_zone_list)
                x2 = [-1000] * len(atom_plane_zone_list)
                y2 = [-1000] * len(atom_plane_zone_list)
                x1[i] = atom_plane.atom_list[j].pixel_x * scale
                y1[i] = atom_plane.atom_list[j].pixel_y * scale
                x2[i] = atom_plane.atom_list[j + 1].pixel_x * scale
                y2[i] = atom_plane.atom_list[j + 1].pixel_y * scale
                marker = LineSegment(x1=x1, y1=y1, x2=x2, y2=y2, color=color)
                marker_list.append(marker)
            if add_numbers:
                x = [-1000] * len(atom_plane_zone_list)
                y = [-1000] * len(atom_plane_zone_list)
                x[i] = atom_plane.atom_list[0].pixel_x * scale
                y[i] = atom_plane.atom_list[0].pixel_y * scale
                marker = Text(
                    x=x,
                    y=y,
                    text=str(index_atom_plane),
                    color=color,
                    va="top",
                    ha="right",
                )
                marker_list.append(marker)
    return marker_list


def _make_single_atom_plane_marker_list(atom_plane, scale=1.0, color="red"):
    marker_list = []
    for i in range(len(atom_plane.atom_list[1:])):
        x1 = atom_plane.atom_list[i].pixel_x * scale
        y1 = atom_plane.atom_list[i].pixel_y * scale
        x2 = atom_plane.atom_list[i + 1].pixel_x * scale
        y2 = atom_plane.atom_list[i + 1].pixel_y * scale
        marker = LineSegment(x1=x1, y1=y1, x2=x2, y2=y2, color=color)
        marker_list.append(marker)
    return marker_list


def _make_arrow_marker_list(arrow_data_list, scale=1.0, color="red"):
    marker_list = []
    for arrow_data in arrow_data_list:
        x, y, vecX, vecY = arrow_data
        marker = _make_single_marker_arrow(x, y, vecX, vecY, scale=scale, color=color)
        marker_list.append(marker)
    return marker_list


def _make_single_marker_arrow(x, y, vecX, vecY, scale=1.0, color="red"):
    x1, x2 = x + vecX / 2, x - vecX / 2
    y1, y2 = y - vecY / 2, y + vecY / 2
    marker = LineSegment(
        x1=x1 * scale, y1=y1 * scale, x2=x2 * scale, y2=y2 * scale, color=color
    )
    return marker


def vector_list_to_marker_list(vector_list, color="red", scale=1.0):
    """Make a marker list from a vector list.

    Parameters
    ----------
    vector_list : list
        In the form [[x0, y0, dx0, dy0], ...]
    color : string, optional
        Default 'red'
    scale : scalar, optional
        Default 1.

    Returns
    -------
    marker_list : list of markers

    Examples
    --------
    >>> import atomap.plotting as pl
    >>> vector_list = [[13, 11, -2, 1], [20, 12, 2, -3]]
    >>> marker_list = pl.vector_list_to_marker_list(
    ...     vector_list, color='red', scale=1.)

    """
    marker_list = []
    for x, y, dx, dy in vector_list:
        x1, y1 = x - dx, y - dy
        marker = LineSegment(x * scale, y * scale, x1 * scale, y1 * scale, color=color)
        marker_list.append(marker)
    return marker_list


def _make_zone_vector_text_marker_list(
    zone_vector_list, x=1, y=1, scale=1.0, color="red"
):
    number = len(zone_vector_list)
    marker_list = []
    if len(zone_vector_list) == 1:
        marker_list.append(
            Text(x, y, text=str(zone_vector_list[0]), size=20, color=color)
        )
    else:
        for index, zone_vector in enumerate(zone_vector_list):
            xP, yP = [-1000] * number, [-1000] * number
            xP[index] = x
            yP[index] = y
            marker = Text(xP, yP, text=str(zone_vector), size=20, color=color)
            marker_list.append(marker)
    return marker_list
