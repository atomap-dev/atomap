import numpy as np
import math
import atomap.tools as to
import hyperspy.api as hs


def get_neighbor_middle_position(atom, za0, za1):
    """Find the middle point between four neighboring atoms.

    The neighbors are found by moving one step along the atom planes
    belonging to za0 and za1.

    So atom planes must be constructed first.

    Parameters
    ----------
    atom : Atom_Position object
    za0 : tuple
    za1 : tuple

    Return
    ------
    middle_position : tuple
        If the atom is at the edge by being the last atom in the
        atom plane, False is returned.

    Examples
    --------
    >>> import atomap.analysis_tools as an
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.construct_zone_axes()
    >>> za0 = sublattice.zones_axis_average_distances[0]
    >>> za1 = sublattice.zones_axis_average_distances[1]
    >>> atom = sublattice.atom_list[33]
    >>> middle_position = an.get_neighbor_middle_position(atom, za0, za1)

    """
    atom00 = atom
    atom01 = atom.get_next_atom_in_zone_vector(za0)
    atom10 = atom.get_next_atom_in_zone_vector(za1)
    middle_position = False
    if not (atom01 is False):
        if not (atom10 is False):
            atom11 = atom10.get_next_atom_in_zone_vector(za0)
            if not (atom11 is False):
                middle_position = to.get_point_between_four_atoms(
                    (atom00, atom01, atom10, atom11)
                )
    return middle_position


def get_middle_position_list(sublattice, za0, za1):
    """Find the middle point between all four neighboring atoms.

    The neighbors are found by moving one step along the atom planes
    belonging to za0 and za1.

    So atom planes must be constructed first.

    Parameters
    ----------
    sublattice : Sublattice object
    za0 : tuple
    za1 : tuple

    Return
    ------
    middle_position_list : list

    Examples
    --------
    >>> import atomap.analysis_tools as an
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.construct_zone_axes()
    >>> za0 = sublattice.zones_axis_average_distances[0]
    >>> za1 = sublattice.zones_axis_average_distances[1]
    >>> middle_position_list = an.get_middle_position_list(
    ...     sublattice, za0, za1)

    """
    middle_position_list = []
    for atom in sublattice.atom_list:
        middle_pos = get_neighbor_middle_position(atom, za0, za1)
        if not (middle_pos is False):
            middle_position_list.append(middle_pos)
    return middle_position_list


def get_vector_shift_list(sublattice, position_list):
    """Find the atom shifts from a central position.

    Useful for finding polarization in B-cations in a perovskite structure.

    Parameters
    ----------
    sublattice : Sublattice object
    position_list : list
        [[x0, y0], [x1, y1], ...]

    Returns
    -------
    vector_list : list
        In the form [[x0, y0, dx0, dy0], [x1, y1, dx1, dy1]...]

    Example
    -------
    >>> import atomap.analysis_tools as an
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.construct_zone_axes()
    >>> za0 = sublattice.zones_axis_average_distances[0]
    >>> za1 = sublattice.zones_axis_average_distances[1]
    >>> middle_position_list = an.get_middle_position_list(
    ...     sublattice, za0, za1)
    >>> vector_list = an.get_vector_shift_list(
    ...     sublattice, middle_position_list)

    """
    vector_list = []
    for position in position_list:
        dist = np.hypot(
            np.array(sublattice.x_position) - position[0],
            np.array(sublattice.y_position) - position[1],
        )
        atom_b = sublattice.atom_list[dist.argmin()]
        vector = (
            position[0],
            position[1],
            position[0] - atom_b.pixel_x,
            position[1] - atom_b.pixel_y,
        )
        vector_list.append(vector)
    return vector_list


def pair_distribution_function(image, atom_positions, n_bins=200, rel_range=0.5):
    """
    Returns a two dimensional pair distribution function (PDF) from an image of
    atomic columns.

    The intensity of peaks in the PDF is corrected to account for missing
    information (i.e. the fact atoms are present outside of the field of view)
    and differences in area at different distances.

    Parameters
    ----------
    image : 2D HyperSpy Signal object
    atom_positions : NumPy array
        A NumPy array of [x, y] atom positions.
    n_bins : int
        Number of bins to use for the PDF.
    rel_range : float
        The range of the PDF as a fraction of the field of view of the image.

    Returns
    -------
    s_pdf : HyperSpy Signal 1D Object
        The calculated PDF.

    Examples
    --------
    >>> import atomap.analysis_tools as an
    >>> s = am.dummy_data.get_simple_cubic_signal()
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> s_pdf = an.pair_distribution_function(s,sublattice.atom_positions)
    >>> s_pdf.plot()

    """
    pair_distances = []
    distance_from_edge = []

    x_size = image.axes_manager[0].size
    y_size = image.axes_manager[1].size
    x_scale = image.axes_manager[0].scale
    y_scale = image.axes_manager[1].scale

    if isinstance(image.axes_manager[0].units, str):
        units = image.axes_manager[0].units
    else:
        units = "pixels"

    for i, position1 in enumerate(atom_positions):
        dist_edge_x = min([position1[0] * x_scale, (x_size - position1[0]) * x_scale])
        dist_edge_y = min([position1[1] * y_scale, (y_size - position1[1]) * y_scale])
        distance_from_edge.append([dist_edge_x, dist_edge_y])
        for position2 in atom_positions[i:]:
            if not np.array_equal(position1, position2):
                dist_x = position1[0] * x_scale - position2[0] * x_scale
                dist_y = position1[1] * y_scale - position2[1] * y_scale
                pair_distance = math.hypot(dist_x, dist_y)
                pair_distances.append(pair_distance)

    intensities, bins = np.histogram(
        pair_distances, bins=n_bins, range=(0, rel_range * min([x_size, y_size]))
    )

    intensities = intensities.astype(float)
    for i, intensity in enumerate(intensities):
        intensity = 2 * intensity / len(atom_positions)
        area_correction = []
        for distance in distance_from_edge:
            if min(distance) < bins[i + 1]:
                area_correction.append(1 - _area_proportion(distance, bins[i + 1]))
            else:
                area_correction.append(1)
        if len(area_correction) > 0:
            mean = sum(area_correction) / len(area_correction)
        else:
            mean = 1
        intensities[i] = intensity / mean

    axis_dict = {
        "name": "r",
        "units": units,
        "scale": bins[1],
        "size": len(intensities),
    }
    intensity_signal = hs.signals.Signal1D(intensities, axes=[axis_dict])
    intensity_signal.metadata.General.title = "Pair distribution function"
    return intensity_signal


def _area_proportion(distances, radius):
    distances = -(distances - radius)
    dist_norm = [(i > 0) * i for i in distances]
    if 0 in dist_norm:
        proportion = math.acos(max(dist_norm) / radius) / math.pi
    else:
        proportion = 0.25 + (
            math.acos(dist_norm[0] / radius) + math.acos(dist_norm[1] / radius)
        ) / (2 * math.pi)
    return proportion
