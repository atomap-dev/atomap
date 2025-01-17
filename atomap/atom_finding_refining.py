from hyperspy.external.progressbar import progressbar
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import hyperspy.api as hs
from hyperspy._signals.signal2d import Signal2D
from hyperspy.drawing._markers.points import Points
import numpy as np
from skimage.feature import peak_local_max
from sklearn.decomposition import TruncatedSVD
import math
import warnings

from atomap.external.gaussian2d import Gaussian2D


def get_atom_positions(
    signal,
    separation=5,
    threshold_rel=0.02,
    pca=False,
    subtract_background=False,
    normalize_intensity=False,
    remove_too_close_atoms=True,
):
    """
    Find the most intense features in a HyperSpy signal, where the
    features has to be separated by a minimum distance.

    Parameters
    ----------
    signal : HyperSpy 2D signal
    separation : number
        Minimum separation between the features.
    threshold_rel : float, default 0.02
    pca : bool, default False
        Do PCA on the signal before doing the peak finding.
    subtract_background : bool, default False
        Subtract the average background from the signal before
        doing the peak finding.
    normalize_intensity : bool, default False
    remove_too_close_atoms : bool, default True
        Will attempt to find and remove atoms which are too close to
        eachother, i.e. less than separation.

    Returns
    -------
    NumPy array, list of the most intense atom positions.

    Example
    -------
    >>> import numpy as np
    >>> import atomap.api as am
    >>> s = am.dummy_data.get_simple_cubic_signal()
    >>> from atomap.atom_finding_refining import get_atom_positions
    >>> atom_positions = get_atom_positions(s, 5)
    >>> peak_x = atom_positions[:,0]
    >>> peak_y = atom_positions[:,1]
    """
    if separation < 1:
        raise ValueError("Separation can not be smaller than 1")
    sig_dims = len(signal.data.shape)
    if sig_dims != 2:
        raise ValueError("signal must have 2 dimensions, not {0}".format(sig_dims))
    if pca:
        signal = do_pca_on_signal(signal)
    if subtract_background:
        signal = subtract_average_background(signal)
    if normalize_intensity:
        signal = normalize_signal(signal)

    image_data = signal.data
    if image_data.dtype is np.dtype("float16"):
        image_data = image_data.astype("float32")
    if image_data.dtype is np.dtype("int8"):
        image_data = image_data.astype("int32")
    if image_data.dtype is np.dtype("int16"):
        image_data = image_data.astype("int32")

    temp_positions = peak_local_max(
        image=image_data,
        min_distance=int(separation),
        threshold_rel=threshold_rel,
    )

    # The X- and Y-axes are switched in HyperSpy compared to NumPy
    # so we need to flip them here
    atom_positions = np.fliplr(temp_positions)
    if remove_too_close_atoms:
        atom_positions = _remove_too_close_atoms(atom_positions, int(separation) / 2)
    return atom_positions


def _remove_too_close_atoms(
    atom_positions, pixel_separation_tolerance, intensities=None, max_iter=20
):
    """Remove atoms which are within the tolerance from a list of positions

    Parameters
    ----------
    atom_positions : NumPy array
        In the form [[x0, y0], [x1, y1], ...]
    pixel_separation_tolerance : scalar
        Minimum separation between the positions.
    intensities : NumPy array, optional
        An intensity value corresponding to each atom_position to decide
        which atom is more appropriate to remove in a pair. If not provided it will be
        assumed that peaks are sorted from most to least intense.
    max_iter : int, optional
        Maximum number of iterations of removing points as a back-stop against
        infinite loop.

    Returns
    -------
    atom_positions_new : NumPy array
        With the too close atoms removed.

    Examples
    --------
    >>> import atomap.atom_finding_refining as afr
    >>> data = np.array([[1, 10], [10, 1], [4, 10]])
    >>> data_new = afr._remove_too_close_atoms(data, 5)

    """
    if intensities is None:
        intensities = np.arange(atom_positions.shape[0])[::-1]
    i = 0
    converged = False
    while i < max_iter:
        tree = cKDTree(atom_positions)
        pairs = tree.query_pairs(pixel_separation_tolerance)
        if not pairs:
            converged = True
            break
        pairs_ar = np.array(list(pairs))
        pair_intensities = intensities[pairs_ar]
        min_int_col = np.argmin(pair_intensities, axis=1)
        # Index of each pair where intensity is lowest and highest resp
        minimum_int_indexes = pairs_ar[np.arange(min_int_col.shape[0]), min_int_col]
        maximum_int_indexes = pairs_ar[np.arange(min_int_col.shape[0]), 1 - min_int_col]
        # Only consider pairs where the maximum index is not in the minimum index;
        # those rows would be removed anyway
        original_tuples = np.where(
            np.isin(maximum_int_indexes, minimum_int_indexes) == False
        )[0]
        # The rows corresponding to unique indices in the minimum column must be removed
        remove_indexes = np.unique(minimum_int_indexes[original_tuples])
        atom_positions = np.delete(atom_positions, remove_indexes, axis=0)
        # we must iterate to check for non-resolved situations e.g. triplets
        i += 1
    if not converged:
        warnings.warn(
            f"Not all additional atoms could be removed in {max_iter} iterations!"
        )
    return atom_positions


def find_features_by_separation(
    signal,
    separation_range,
    separation_step=1,
    threshold_rel=0.02,
    pca=False,
    subtract_background=False,
    normalize_intensity=False,
    show_progressbar=True,
):
    """
    Do peak finding with a varying amount of peak separation
    constrained.

    Inspiration from the program Smart Align by Lewys Jones.

    Parameters
    ----------
    signal : HyperSpy 2D signal
    separation_range : tuple
        Lower and upper end of minimum pixel distance between the
        features.
    separation_step : int, optional
    show_progressbar : bool, default True

    Returns
    -------
    tuple, (separation_list, peak_list)

    """
    separation_list = range(separation_range[0], separation_range[1], separation_step)

    separation_value_list = []
    peak_list = []
    for separation in progressbar(separation_list, disable=not show_progressbar):
        peaks = get_atom_positions(
            signal,
            separation=separation,
            threshold_rel=threshold_rel,
            pca=pca,
            normalize_intensity=normalize_intensity,
            subtract_background=subtract_background,
        )

        separation_value_list.append(separation)
        peak_list.append(peaks)

    return (separation_value_list, peak_list)


def get_feature_separation(
    signal,
    separation_range=(5, 30),
    separation_step=1,
    pca=False,
    subtract_background=False,
    normalize_intensity=False,
    threshold_rel=0.02,
    show_progressbar=True,
):
    """
    Plot the peak positions on in a HyperSpy signal, as a function
    of peak separation.

    Parameters
    ----------
    signal : HyperSpy signal 2D
    separation_range : tuple, optional, default (5, 30)
    separation_step : int, optional, default 1
    pca : bool, default False
    subtract_background : bool, default False
    normalize_intensity : bool, default False
    threshold_rel : float, default 0.02
    show_progressbar : bool, default True

    Example
    -------
    >>> import numpy as np
    >>> import hyperspy.api as hs
    >>> from atomap.atom_finding_refining import get_feature_separation
    >>> s = hs.signals.Signal2D(np.random.random((500, 500)))
    >>> s1 = get_feature_separation(s, show_progressbar=False)

    """
    if separation_range[0] > separation_range[1]:
        raise ValueError(
            "The lower range of the separation_range ({0}) can not be "
            "smaller than the upper range ({1})".format(
                separation_range[0], separation_range[0]
            )
        )
    if separation_range[0] < 1:
        raise ValueError(
            "The lower range of the separation_range can not be below 1. "
            "Current value: {0}".format(separation_range[0])
        )

    if signal.data.dtype is np.dtype("float16"):
        raise ValueError(
            "signal has dtype float16, which is not supported "
            "use signal.change_dtype('float32') to change it"
        )

    separation_list, peak_list = find_features_by_separation(
        signal=signal,
        separation_range=separation_range,
        separation_step=separation_step,
        threshold_rel=threshold_rel,
        pca=pca,
        normalize_intensity=normalize_intensity,
        subtract_background=subtract_background,
        show_progressbar=show_progressbar,
    )

    scale_x = signal.axes_manager[0].scale
    scale_y = signal.axes_manager[1].scale
    offset_x = signal.axes_manager[0].offset
    offset_y = signal.axes_manager[1].offset

    s = hs.stack([signal] * len(separation_list), show_progressbar=show_progressbar)
    s.axes_manager.navigation_axes[0].offset = separation_list[0]
    s.axes_manager.navigation_axes[0].scale = separation_step
    s.axes_manager.navigation_axes[0].name = "Feature separation, [Pixels]"
    s.axes_manager.navigation_axes[0].unit = "Pixels"

    max_peaks = 0
    for peaks in peak_list:
        if len(peaks) > max_peaks:
            max_peaks = len(peaks)
    if max_peaks == 0:
        raise ValueError(
            "No peaks found, try either reducing separation_range, or "
            "using a better image"
        )

    offsets = np.empty(len(peak_list), dtype=object)
    for ind in range(len(peak_list)):
        offsets[ind] = peak_list[ind] * [scale_x, scale_y] + [offset_x, offset_y]

    p = Points(offsets=offsets, color="red", sizes=4)
    s.add_marker(p, permanent=True, plot_marker=False)
    return s


def find_feature_density(
    image_data,
    separation_range=None,
    separation_step=1,
    plot_figure=False,
    plot_debug_figures=False,
):
    """
    Do peak finding with a varying amount of peak separation
    constrained. Gives a measure of feature density, and
    what peak separation should be used to find the initial
    sub-lattice.

    Inspiration from the program Smart Align by Lewys Jones.
    """

    separation_list, peak_list = find_features_by_separation(
        image_data=image_data,
        separation_range=separation_range,
        separation_step=separation_step,
    )
    peakN_list = []
    for peaks in peak_list:
        peakN_list.append(len(peaks))

    return (separation_list, peakN_list)


def construct_zone_axes_from_sublattice(
    sublattice,
    atom_plane_tolerance=0.5,
    zone_axis_para_list=False,
    nearest_neighbors=15,
):
    """Constructs zone axes for a sublattice.

    The zone axes are constructed by finding the 15 nearest neighbors for
    each atom position in the sublattice, and finding major translation
    symmetries among the nearest neighbours. Only unique zone axes are kept,
    and "bad" ones are removed.

    Parameters
    ----------
    sublattice : Atomap Sublattice
    atom_plane_tolerance : scalar, default 0.5
        When constructing the atomic planes, the method will try to locate
        the atoms by "jumping" one zone vector, and seeing if there is an atom
        with the pixel_separation times atom_plane_tolerance. So this value
        should be increased the atomic planes are non-continuous and "split".
    zone_axis_para_list : parameter list or bool, default False
        A zone axes parameter list is used to name and index the zone axes.
        See atomap.process_parameters for more info. Useful for automation.
    nearest_neighbors : int, default 15
        The number of nearest neighbors which are calculated for each
        atomic position. Increase this if you want to get more
        atomic planes.

    Example
    -------
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice
    <Sublattice,  (atoms:400,planes:0)>
    >>> import atomap.atom_finding_refining as afr
    >>> afr.construct_zone_axes_from_sublattice(sublattice)
    >>> sublattice
    <Sublattice,  (atoms:400,planes:4)>

    Increasing nearest_neighbors to get more planes
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> afr.construct_zone_axes_from_sublattice(sublattice, nearest_neighbors=25)
    >>> sublattice
    <Sublattice,  (atoms:400,planes:8)>


    See also
    --------
    sublattice._make_translation_symmetry : How unique zone axes are found
    sublattice._remove_bad_zone_vectors : How fragmented ("bad zone axis")
        are identified and removed.
    sublattice.construct_zone_axes
    atomap.process_parameters : more info on zone axes parameter list

    """
    if sublattice._pixel_separation == 0.0:
        sublattice._pixel_separation = sublattice._get_pixel_separation()
    sublattice.find_nearest_neighbors(nearest_neighbors=nearest_neighbors)
    sublattice._make_translation_symmetry()

    if zone_axis_para_list is not False:
        zone_axes = []
        zone_axes_names = []
        for zone_axis_para in zone_axis_para_list:
            if zone_axis_para["number"] < len(sublattice.zones_axis_average_distances):
                index = zone_axis_para["number"]
                zone_axes.append(sublattice.zones_axis_average_distances[index])
                zone_axes_names.append(zone_axis_para["name"])
        sublattice.zones_axis_average_distances = zone_axes
        sublattice.zones_axis_average_distances_names = zone_axes_names

    sublattice._generate_all_atom_plane_list(atom_plane_tolerance=atom_plane_tolerance)
    sublattice._sort_atom_planes_by_zone_vector()
    sublattice._remove_bad_zone_vectors()


def _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius):
    """
    Make a circular mask in a bool array for masking a region in an image.

    Parameters
    ----------
    centreX, centreY : float
        Centre point of the mask.
    imageSizeX, imageSizeY : int
        Size of the image to be masked.
    radius : float
        Radius of the mask.

    Returns
    -------
    Boolean Numpy 2D Array
        Array with the shape (imageSizeX, imageSizeY) with the mask.

    See also
    --------
    _make_mask_from_positions

    Examples
    --------
    >>> import numpy as np
    >>> from atomap.atom_finding_refining import _make_circular_mask
    >>> image = np.ones((9, 9))
    >>> mask = _make_circular_mask(4, 4, 9, 9, 2)
    >>> image_masked = image*mask
    >>> import matplotlib.pyplot as plt
    >>> cax = plt.imshow(image_masked)
    """
    y, x = np.ogrid[-centerX : imageSizeX - centerX, -centerY : imageSizeY - centerY]
    mask = x * x + y * y <= radius * radius
    return mask


def _make_mask_circle_centre(arr, radius):
    """Create a circular mask with same shape as arr

    The circle is centered on the center of the array,
    with the circle having False values.

    Similar to _make_circular_mask, but simpler and potentially
    faster.

    Numba jit compatible.

    Parameters
    ----------
    arr : NumPy array
        Must be 2 dimensions
    radius : scalar
        Radius of the circle

    Returns
    -------
    mask : NumPy array
        Boolean array

    Example
    -------
    >>> import atomap.atom_finding_refining as afr
    >>> arr = np.random.randint(100, size=(20, 20))
    >>> mask = afr._make_mask_circle_centre(arr, 10)

    """
    if len(arr.shape) != 2:
        raise ValueError("arr must be 2D, not {0}".format(len(arr.shape)))
    imageSizeX, imageSizeY = arr.shape
    centerX = (arr.shape[0] - 1) / 2
    centerY = (arr.shape[1] - 1) / 2

    x = np.expand_dims(np.arange(-centerX, imageSizeX - centerX), axis=1)
    y = np.arange(-centerY, imageSizeY - centerY)
    mask = x * x + y * y > radius * radius
    return mask


def zero_array_outside_circle(arr, radius):
    """Set all values in an array to zero outside a circle defined by radius

    Numba jit compatible

    Parameters
    ----------
    arr : NumPy array
        Must be 2 dimensions
    radius : scalar
        Radius of the circle

    Returns
    -------
    output_array : NumPy array
        Same shape as arr, but with values outside the circle set to zero.

    Example
    -------
    >>> import atomap.atom_finding_refining as afr
    >>> arr = np.random.randint(100, size=(30, 20))

    """
    shape = arr.shape
    mask = _make_mask_circle_centre(arr, radius).flatten()
    arr = arr.flatten()
    arr[mask] = 0
    return np.reshape(arr, shape)


def _crop_array(arr, center_x, center_y, radius):
    """Crop an array around a center point to give a square.

    The square has sidelengths `2*radius-1`.

    If the center point is such that the radius will intersect the array
    edges, the space outside the array will be padded as zeros.

    Parameters
    ----------
    arr : Numpy 2D Array
    centre_x, centre_y : int
        Centre point of the cropped array.
    radius : int
        Radius of the crop around the center point.

    Returns
    -------
    Numpy 2D Array
        Array with the shape (2*radius-1, 2*radius-1).

    Examples
    --------
    >>> import atomap.atom_finding_refining as afr
    >>> arr = np.random.randint(100, size=(30, 50))
    >>> data = afr._crop_array(arr, 25, 10, 5)

    """
    radius_left = radius - 1
    radius_right = radius

    # Reversed first two indices so we can subtract the edges
    edges_of_crop = np.array(
        [
            radius_left - center_x,
            radius_left - center_y,
            center_x + radius_right,
            center_y + radius_right,
        ]
    )
    ymax, xmax = arr.shape
    edges_of_arr = np.array([0, 0, xmax - 1, ymax - 1])
    edge_difference_max = np.max(edges_of_crop - edges_of_arr)

    if edge_difference_max > 0:
        arr = _pad_array(arr, edge_difference_max)
        center_x += edge_difference_max
        center_y += edge_difference_max
    ymin = center_y - radius_left
    ymax = center_y + radius_right
    xmin = center_x - radius_left
    xmax = center_x + radius_right
    return arr[ymin:ymax, xmin:xmax]


def calculate_center_of_mass(arr):
    """Find the center of mass of a NumPy array.

    Find the center of mass of a single 2D array, or a
    list (or multidimensional array) of 2D arrays.

    Parameters
    ----------
    arr : Numpy 2D Array (or list/nd-array of such)

    Returns
    -------
    cy, cx: array of floats (or nd-array of floats)
        Giving centre coordinates with sub-pixel accuracy

    Examples
    --------
    >>> import atomap.atom_finding_refining as afr
    >>> arr = np.random.randint(100, size=(10, 10))
    >>> data = afr.calculate_center_of_mass(arr)

    Notes
    -----
    This is a much simpler center of mass approach than the one from scipy.
    Gotten from stackoverflow:
    https://stackoverflow.com/questions/37519238/python-find-center-of-object-in-an-image

    """
    # Can consider subtracting minimum value
    # this gives the center of mass higher "contrast"
    # arr -= arr.min()
    if len(arr.shape) > 2:
        arr = (arr.T / np.sum(arr, axis=(-1, -2)).T).T
    else:
        arr = arr / np.sum(arr, axis=(-1, -2))

    dy = np.sum(arr, -1)
    dx = np.sum(arr, -2)

    (Y, X) = arr.shape[-2:]
    cx = np.sum(dx * np.arange(X), axis=-1).T
    cy = np.sum(dy * np.arange(Y), axis=-1).T
    return np.array([cy, cx]).T


def _pad_array(arr, padding=1):
    """Pad an array to give it extra zero-value pixels around the edges.

    Parameters
    ----------
    arr : Numpy 2D Array
    padding : int, optional
        Default 1

    Returns
    -------
    arr2 : NumPy array

    Examples
    --------
    >>> import atomap.atom_finding_refining as afr
    >>> arr = np.random.randint(100, size=(10, 10))
    >>> data = afr._pad_array(arr, padding=2)

    """
    x, y = arr.shape
    arr2 = np.zeros((x + padding * 2, y + padding * 2))
    arr2[padding:-padding, padding:-padding] = arr.copy()
    return arr2


def _make_mask_from_positions(position_list, radius_list, data_shape):
    """
    Parameters
    ----------
    position_list : list of list
        [[x0, y0], [x1, y1]]
    radius_list : list
    data_shape : tuple

    Examples
    --------
    >>> from atomap.atom_finding_refining import _make_mask_from_positions
    >>> pos = [[10, 20], [25, 10]]
    >>> radius = [2, 1]
    >>> mask = _make_mask_from_positions(pos, radius, (40, 40))
    """
    if len(position_list) != len(radius_list):
        raise ValueError("position_list and radius_list must be the same length")
    mask = np.zeros(data_shape, dtype=bool)
    for position, radius in zip(position_list, radius_list):
        mask += _make_circular_mask(
            position[0], position[1], data_shape[0], data_shape[1], radius
        )
    return mask


def _find_background_value(data, method="median", lowest_percentile=0.1):
    """
    Get background value for image data. Intended for Gaussian
    shaped image, with the intensity peak in the middle.
    The median method is preferred to avoid any large outliers
    affecting the background value. Since various acquisition
    artifacts can lead to very low or zero pixel values.

    Parameters
    ----------
    data : Numpy array
    method : 'median', 'mean' or 'minimum'
        If median, the median of the lowest_percentile will
        be used as background. This is the default.
        If mean, the mean of the lowest_percentile is
        used as background.
        If minimum, the lowest value in data will be
        used as background.
    lowest_percentile : number between 0.01 and 1
        The lowest percentile will be used to calculate the
        background, from 1 (0.01) to 100% (1.0). Default 10%

    Returns
    -------
    low_value : float
    """
    if not ((lowest_percentile >= 0.01) and (lowest_percentile <= 1.0)):
        raise ValueError("lowest_percentile must be between 0.01 and 1.0")
    if method == "minimum":
        background_value = data.min()
    else:
        amount = int(lowest_percentile * data.size)
        if amount == 0:
            amount = 1
        lowest_values = np.sort(data.flatten())[:amount]
        if method == "median":
            background_value = np.median(lowest_values)
        elif method == "mean":
            background_value = np.mean(lowest_values)
        else:
            raise ValueError("method must be 'minimum', 'median' or 'mean'")
    return background_value


def _find_median_upper_percentile(data, upper_percentile=0.1):
    """
    Get the median of the upper percentile of an image.
    Useful for finding a good initial max value in an image for doing
    2D Gaussian fitting.
    Median is used to avoid acquisition artifacts leading to too
    high pixel values giving bad initial values.

    Parameters
    ----------
    data : Numpy array
    upper_percentile : number between 0.01 and 1
        The upper percentile will be used to calculate the
        background, from 1 (0.01) to 100% (1.0). Default 10% (0.1).

    Returns
    -------
    high_value : float
    """
    if not ((upper_percentile >= 0.01) and (upper_percentile <= 1.0)):
        raise ValueError("lowest_percentile must be between 0.01 and 1.0")
    amount = int(upper_percentile * data.size)
    if amount == 0:
        amount = 1
    high_value = np.median(np.sort(data.flatten())[-amount:])
    return high_value


def _atom_to_gaussian_component(atom):
    """
    Make a HyperSpy 2D gaussian component from an Atomap Atom_Position.

    Parameter
    ---------
    atom : Atomap Atom_Position object

    Return
    ------
    HyperSpy 2D gaussian component

    Example
    -------
    >>> from atomap.atom_position import Atom_Position
    >>> from atomap.atom_finding_refining import _atom_to_gaussian_component
    >>> atom = Atom_Position(x=5.2, y=7.7, sigma_x=2.1, sigma_y=1.1)
    >>> gaussian = _atom_to_gaussian_component(atom)
    """
    g = Gaussian2D(
        centre_x=atom.pixel_x,
        centre_y=atom.pixel_y,
        sigma_x=atom.sigma_x,
        sigma_y=atom.sigma_y,
        rotation=atom.rotation,
    )
    return g


def _copy_gaussian_parameters_to_atom(atom, g):
    """
    Copy HyperSpy 2D Gaussian parameters to an Atomap Atom_Position.

    Parameter
    ---------
    atom : Atomap Atom_Position object
    g : HyperSpy 2D Gaussian component

    """

    atom.old_pixel_x_list.append(atom.pixel_x)
    atom.old_pixel_y_list.append(atom.pixel_y)
    atom.pixel_x = g.centre_x.value
    atom.pixel_y = g.centre_y.value
    atom.rotation = g.rotation.value % math.pi
    atom.sigma_x = abs(g.sigma_x.value)
    atom.sigma_y = abs(g.sigma_y.value)
    atom.amplitude_gaussian = g.A.value
    atom._gaussian_fitted = True


def get_mask_slice(position_list, radius_list):
    """Get a NumPy slice for extracting atom positions and
    the region around them.

    Parameters
    ----------
    position_list : list of tuples
        Positions of the atom positions, (y, x)
    radius_list : list of numbers
        Region around the atoms

    Returns
    -------
    NumPy 2D slice

    """
    mask_position_list_x = []
    mask_position_list_y = []
    for position, radius in zip(position_list, radius_list):
        mask_position_list_y.append(position[0] - radius)
        mask_position_list_y.append(position[0] + radius)
        mask_position_list_x.append(position[1] - radius)
        mask_position_list_x.append(position[1] + radius)
    mask_min_x = int(round(min(mask_position_list_x)))
    mask_max_x = int(round(max(mask_position_list_x))) + 1
    mask_min_y = int(round(min(mask_position_list_y)))
    mask_max_y = int(round(max(mask_position_list_y))) + 1
    if mask_min_x < 0:
        mask_min_x = 0
    if mask_min_y < 0:
        mask_min_y = 0

    mask_slice = np.s_[mask_min_y:mask_max_y, mask_min_x:mask_max_x]
    return mask_slice


def _region_around_atoms_as_signal(
    atom_list,
    image_data,
    mask_radius=None,
    percent_to_nn=0.4,
):
    """
    Get a subset of an image, based on atom positions and fitting
    region around each atom.

    Parameters
    ----------
    atom_list : list of Atomap Atom_Position objects
        This can also be a single Atom_Position.
    image_data : NumPy 2D array
    mask_radius : scalar
        Default None
    percent_to_nn : float
        Default 0.4

    Returns
    -------
    signal_crop : HyperSpy 2D Signal
        signal_crop.axes_manager[...].offset is used to set the correct
        location from where the signal is extracted from image_data
    shifted_pos_list : list of tuples
    radius_list : list of numbers

    """
    position_list, radius_list = [], []
    for atom in atom_list:
        position_list.append((atom.pixel_y, atom.pixel_x))
        if mask_radius is None:
            mask_radius = atom.get_closest_neighbor() * percent_to_nn
        radius_list.append(mask_radius)

    mask_slice = get_mask_slice(position_list, radius_list)
    mask_min_y = mask_slice[0].start
    mask_min_x = mask_slice[1].start
    image_data_crop = image_data[mask_slice]

    signal_crop = Signal2D(image_data_crop)
    signal_crop.axes_manager[0].offset = mask_min_x
    signal_crop.axes_manager[1].offset = mask_min_y

    shifted_pos_list = []
    for position in position_list:
        shifted_position = (position[0] - mask_min_y, position[1] - mask_min_x)
        shifted_pos_list.append(shifted_position)

    return signal_crop, shifted_pos_list, radius_list


def _make_model_from_atom_list(atom_list, signal_crop):
    """
    Make a HyperSpy model from a list of Atom_Position objects and
    an HyperSpy Signal2D object.

    Parameters
    ----------
    atom_list : list of Atom_Position objects
        List of atoms to be included in the model.
    signal_crop : HyperSpy Signal2D

    Returns
    -------
    model : HyperSpy model
        Model where the atoms are added as gaussian components.

    See also
    --------
    _fit_atom_positions_with_gaussian_model
    fit_atom_positions_gaussian

    Examples
    --------
    >>> import numpy as np
    >>> import hyperspy.api as hs
    >>> from atomap.atom_position import Atom_Position
    >>> from atomap.atom_finding_refining import _make_model_from_atom_list
    >>> atom_list = [Atom_Position(2, 2), Atom_Position(4, 4)]
    >>> image = np.random.random((100, 100))
    >>> s = hs.signals.Signal2D(image)
    >>> m = _make_model_from_atom_list(
    ...     atom_list=atom_list, signal_crop=s)
    >>> m.fit(print_info=False, return_info=False)

    """
    upper_value = _find_median_upper_percentile(signal_crop.data, upper_percentile=0.03)
    gaussian_list = []
    for atom in atom_list:
        gaussian = _atom_to_gaussian_component(atom)
        if atom._gaussian_fitted:
            gaussian.A.value = atom.amplitude_gaussian
        else:
            gaussian.A.value = upper_value * 10
        gaussian_list.append(gaussian)

    m = signal_crop.create_model()
    m.extend(gaussian_list)
    return m


def _fit_atom_positions_with_gaussian_model(
    model,
    atom_list,
    mask,
    rotation_enabled=True,
    centre_free=True,
):
    """
    Fit a list of Atom_Positions to an image using 2D gaussians.
    This type of fitting is prone to errors, especially where the atoms are
    close together, and on noisy images. To reduce the odds of bad
    fitting this function will return False if:
    - Any of the fitted positions are outside the model region.
    - The sigma ratio (highest_sigma/lowest_sigma) is higher than 4.

    Parameters
    ----------
    model : HyperSpy model object
    atom_list : list of Atom_Position objects
    mask : 2-D NumPy array
    rotation_enabled : bool, optional
        If True (default), the 2D gaussian will be able to rotate.
    centre_free : bool, optional
        If True (default), the gaussian will be free to move. Setting this
        to False can be used to find better values for the other parameters
        like sigma and A, without the centre positions causing bad fitting.

    Returns
    -------
    gaussian_list : list of the fitted gaussians

    See also
    --------
    _make_model_from_atom_list
    fit_atom_positions_gaussian

    """
    # If the model has less than 6 points for fitting, the fitting will
    # not work, so return False
    if model.signal.axes_manager.signal_size < 6:
        return False

    if centre_free is False:
        for g in model:
            g.centre_x.free = False
            g.centre_y.free = False
    for g in model:
        if rotation_enabled:
            g.rotation.free = True
        else:
            g.rotation.free = False

    model.fit()

    gaussian_list = []
    axm_x = model.signal.axes_manager[0]
    axm_y = model.signal.axes_manager[1]
    for atom, g in zip(atom_list, model):
        # If the Gaussian centre is located outside the masked region,
        # return False
        try:
            centre_y = axm_y.value2index(g.centre_y.value)
            centre_x = axm_x.value2index(g.centre_x.value)
        except ValueError:
            # If ValueError is raised the center point is most likely
            # outside the signal itself, and the fitting was bad
            return False
        if not (0 < centre_y < model.signal.data.shape[0]):
            return False
        if not (0 < centre_x < model.signal.data.shape[1]):
            return False
        inside_mask = mask[int(centre_y)][int(centre_x)]
        if not inside_mask:
            return False
        if g.A.value < 0.0:
            return False

        # If sigma aspect ratio is too large, assume the fitting is bad
        max_sigma = max((abs(g.sigma_x.value), abs(g.sigma_y.value)))
        min_sigma = min((abs(g.sigma_x.value), abs(g.sigma_y.value)))
        sigma_ratio = max_sigma / min_sigma
        if sigma_ratio > 4:
            return False
        gaussian_list.append(g)

    return gaussian_list


def fit_atom_positions_gaussian(
    atom_list,
    image_data,
    rotation_enabled=True,
    percent_to_nn=0.40,
    mask_radius=None,
    centre_free=True,
):
    """Fit a list of Atom_Positions to an image using 2D Gaussians.

    The results of the fitting will be saved in the Atom_Position objects
    themselves, and the old positions will be added to
    atom.old_pixel_x_list and atom.old_pixel_y_list.

    This type of fitting is prone to errors, especially where the atoms are
    close together, and on noisy images. To reduce the odds of bad
    fitting the function will rerun the model with a slightly smaller
    mask_radius or percent_to_nn if:
    - Any of the fitted positions are outside the model region.
    - The sigma ratio (highest_sigma/lowest_sigma) is higher than 4.

    This repeats 10 times, and if the fitting is still bad, center of mass
    will be used to set the center position for the atom.

    Parameters
    ----------
    atom_list : list of Atom_Position objects
        For example the members of a dumbbell.
    image_data : NumPy 2D array
    rotation_enabled : bool, optional
        If True (default), the 2D Gaussian will be able to rotate.
    percent_to_nn : float, optional
    mask_radius : float, optional
        Radius of the mask around each atom. If this is not set,
        the radius will be the distance to the nearest atom in the
        same sublattice times the `percent_to_nn` value.
        Note: if `mask_radius` is not specified, the Atom_Position objects
        must have a populated nearest_neighbor_list. This is normally done
        through the sublattice class, but can also be done manually.
        See below for an example how to do this.
    centre_free : bool, optional
        If True (default), the Gaussian will be free to move. Setting this
        to False can be used to find better values for the other parameters
        like sigma and A, without the centre positions causing bad fitting.

    See also
    --------
    _make_model_from_atom_list
    _fit_atom_positions_with_gaussian_model
    atom_lattice.Dumbbell_Lattice for examples on how associated atom
    positions can be fitted together.

    Examples
    --------
    >>> import numpy as np
    >>> from atomap.atom_position import Atom_Position
    >>> from atomap.atom_finding_refining import fit_atom_positions_gaussian

    Fitting atomic columns one-by-one

    >>> atom_list = [Atom_Position(2, 2), Atom_Position(4, 4)]
    >>> image = np.zeros((9, 9))
    >>> for atom_position in atom_list:
    ...     g_list = fit_atom_positions_gaussian(
    ...         atom_list=[atom_position], image_data=image, mask_radius=2)

    Fitting two atoms together

    >>> atom_list = [Atom_Position(2, 2), Atom_Position(4, 4)]
    >>> image = np.zeros((9, 9))
    >>> image[2, 2] = 1.
    >>> image[4, 4] = 1.
    >>> g_list = fit_atom_positions_gaussian(
    ...     atom_list=atom_list, image_data=image, mask_radius=2)

    Not using `mask_radius`, populating the nearest_neighbor_list manually

    >>> image = np.zeros((9, 9))
    >>> image[2, 2] = 1.
    >>> image[5, 5] = 1.
    >>> atom0 = Atom_Position(2, 2, 0.5, 0.5)
    >>> atom1 = Atom_Position(5, 5, 0.5, 0.5)
    >>> atom0.nearest_neighbor_list = [atom1]
    >>> atom1.nearest_neighbor_list = [atom0]
    >>> g_list = fit_atom_positions_gaussian([atom0, atom1], image)

    """
    if (mask_radius is None) and (percent_to_nn is None):
        raise ValueError
    if image_data.dtype != np.dtype("float64"):
        image_data = image_data.astype("float64")

    signal_crop, shifted_pos_list, radius_list = _region_around_atoms_as_signal(
        atom_list=atom_list,
        image_data=image_data,
        mask_radius=mask_radius,
        percent_to_nn=percent_to_nn,
    )

    temp_mask_radius = mask_radius
    temp_percent_to_nn = percent_to_nn
    for i in range(10):
        mask_crop = _make_mask_from_positions(
            shifted_pos_list, radius_list, signal_crop.data.shape
        )
        signal_crop.data = signal_crop.data * mask_crop

        lower_value = _find_background_value(
            signal_crop.data[mask_crop], lowest_percentile=0.03
        )
        signal_crop.data -= lower_value
        signal_crop.data[signal_crop.data < 0] = 0.0

        model = _make_model_from_atom_list(
            atom_list=atom_list,
            signal_crop=signal_crop,
        )

        g_list = _fit_atom_positions_with_gaussian_model(
            model=model,
            atom_list=atom_list,
            mask=mask_crop,
            rotation_enabled=True,
            centre_free=True,
        )

        if g_list is False:
            if i == 9:
                for atom in atom_list:
                    atom.old_pixel_x_list.append(atom.pixel_x)
                    atom.old_pixel_y_list.append(atom.pixel_y)
                    atom.pixel_x, atom.pixel_y = atom._get_center_position_com(
                        image_data,
                        percent_to_nn=temp_percent_to_nn,
                        mask_radius=temp_mask_radius,
                    )
                    atom.amplitude_gaussian = 0.0
                break
            else:
                # If the fitting is bad, reduce the size of the mask, and try again.
                # Bad fitting is often caused by intensity from neighboring atoms,
                # thus reducing the mask size might improve the fitting
                if (temp_mask_radius is None) and (temp_percent_to_nn is None):
                    raise ValueError
                if temp_percent_to_nn is not None:
                    temp_percent_to_nn *= 0.95
                if temp_mask_radius is not None:
                    temp_mask_radius *= 0.95
                radius_list = []
                for atom in atom_list:
                    if mask_radius is None:
                        mask_radius = atom.get_closest_neighbor() * temp_percent_to_nn
                    radius_list.append(mask_radius)
        else:
            for g, atom in zip(g_list, atom_list):
                _copy_gaussian_parameters_to_atom(atom, g)
            break


def refine_sublattice(sublattice, refinement_config_list, percent_to_nn):
    total_number_of_refinements = 0
    for refinement_config in refinement_config_list:
        total_number_of_refinements += refinement_config[1]

    sublattice.find_nearest_neighbors()

    current_counts = 1
    for refinement_config in refinement_config_list:
        image = refinement_config[0]
        number_of_refinements = refinement_config[1]
        refinement_type = refinement_config[2]
        for index in range(1, number_of_refinements + 1):
            print(str(current_counts) + "/" + str(total_number_of_refinements))
            if refinement_type == "gaussian":
                sublattice.refine_atom_positions_using_2d_gaussian(
                    image, rotation_enabled=False, percent_to_nn=percent_to_nn
                )
                sublattice.refine_atom_positions_using_2d_gaussian(
                    image, rotation_enabled=True, percent_to_nn=percent_to_nn
                )
            elif refinement_type == "center_of_mass":
                sublattice.refine_atom_positions_using_center_of_mass(
                    image, percent_to_nn=percent_to_nn
                )
            current_counts += 1


def do_pca_on_signal(signal, pca_components=22):
    signal.change_dtype("float64")
    model = TruncatedSVD(n_components=pca_components)
    # for compatibility with hyperspy's decomposition, transpose
    model.fit(signal.data.T)
    projected = model.transform(signal.data.T)
    truncated = model.inverse_transform(projected).T
    temp_signal = Signal2D(truncated)
    temp_signal.axes_manager[0].scale = signal.axes_manager[0].scale
    temp_signal.axes_manager[1].scale = signal.axes_manager[1].scale
    if temp_signal.data.min() < 0.0:
        temp_signal.data -= temp_signal.data.min()
    return temp_signal


def subtract_average_background(signal, gaussian_blur=30):
    signal.change_dtype("float64")
    temp_signal = signal.deepcopy()
    average_background_data = gaussian_filter(
        temp_signal.data, gaussian_blur, mode="nearest"
    )
    background_subtracted = signal.deepcopy().data - average_background_data
    temp_signal = hs.signals.Signal1D(
        background_subtracted - background_subtracted.min()
    )
    temp_signal.axes_manager[0].scale = signal.axes_manager[0].scale
    temp_signal.axes_manager[1].scale = signal.axes_manager[1].scale
    return temp_signal


def normalize_signal(signal, invert_signal=False):
    temp_signal = signal.deepcopy()
    if invert_signal:
        temp_signal_data = 1.0 / temp_signal.data
    else:
        temp_signal_data = temp_signal.data
    temp_signal_data = temp_signal_data / temp_signal_data.max()
    temp_signal = Signal2D(temp_signal_data)
    temp_signal.axes_manager[0].scale = signal.axes_manager[0].scale
    temp_signal.axes_manager[1].scale = signal.axes_manager[1].scale
    return temp_signal
