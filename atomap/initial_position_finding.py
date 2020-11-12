import copy
import math
from hyperspy.external.progressbar import progressbar
import numpy as np
import matplotlib.pyplot as plt
from atomap.tools import _get_n_nearest_neighbors, Fingerprinter
from atomap.atom_finding_refining import _make_circular_mask, do_pca_on_signal
from atomap.sublattice import Sublattice
from atomap.atom_lattice import Dumbbell_Lattice
import atomap.tools as to
import atomap.gui_classes as gc
from operator import itemgetter
from matplotlib.colors import LogNorm


def find_dumbbell_vector(atom_positions):
    """Find the vector between the atoms in structure with dumbbells.

    For example GaAs-(110).

    Parameters
    ----------
    atom_positions : list of lists
        In the form [[x0, y0], [x1, y1], ...]

    Returns
    -------
    dumbbell_vector : tuple

    Examples
    --------
    >>> import atomap.initial_position_finding as ipf
    >>> s = am.dummy_data.get_dumbbell_signal()
    >>> atom_positions = am.get_atom_positions(s, separation=4)
    >>> dumbbell_vector = ipf.find_dumbbell_vector(atom_positions)

    """
    test = _get_n_nearest_neighbors(atom_positions, 10, leafsize=100)
    fp = Fingerprinter()
    fp.fit(test)
    clusters = fp.cluster_centers_
    clusters_dist = (clusters[:, 0] ** 2 + clusters[:, 1] ** 2) ** 0.5
    sort_zip = zip(list(clusters_dist), clusters[:, 0], clusters[:, 1])
    cluster_dist, cluster_x, cluster_y = zip(*sorted(sort_zip, key=itemgetter(0)))
    vec0 = cluster_x[0], cluster_y[0]
    vec1 = cluster_x[1], cluster_y[1]

    if (abs(vec0[0] + vec1[0]) > 0.1) or (abs(vec0[1] + vec1[1]) > 0.1):
        raise ValueError(
            "Dumbbell vectors should be antiparallel, but are %r and %r" % (vec0, vec1)
        )
    return vec0


def _get_dumbbell_arrays(s, dumbbell_positions, dumbbell_vector, show_progressbar=True):
    """
    Parameters
    ----------
    s : HyperSpy 2D signal
    dumbbell_positions : list of atomic positions
        In the form [[x0, y0], [x1, y1], [x2, y2], ...]
    dumbbell_vector : tuple
    show_progressbar : bool, default True

    Returns
    -------
    Dumbbell lists : tuple of lists

    Examples
    --------
    >>> import atomap.initial_position_finding as ipf
    >>> s = am.dummy_data.get_dumbbell_signal()
    >>> dumbbell_positions = am.get_atom_positions(s, separation=16)
    >>> atom_positions = am.get_atom_positions(s, separation=4)
    >>> dumbbell_vector = ipf.find_dumbbell_vector(atom_positions)
    >>> d0, d1 = ipf._get_dumbbell_arrays(s, dumbbell_positions,
    ...                                   dumbbell_vector)

    """
    next_pos_list0 = []
    next_pos_list1 = []
    for x, y in zip(dumbbell_positions[:, 0], dumbbell_positions[:, 1]):
        next_pos_list0.append([dumbbell_vector[0] + x, dumbbell_vector[1] + y])
        next_pos_list1.append([-dumbbell_vector[0] + x, -dumbbell_vector[1] + y])
    next_pos_list0 = np.array(next_pos_list0)
    next_pos_list1 = np.array(next_pos_list1)

    mask_radius = 0.5 * (dumbbell_vector[0] ** 2 + dumbbell_vector[1] ** 2) ** 0.5

    iterator = zip(
        dumbbell_positions[:, 0],
        dumbbell_positions[:, 1],
        next_pos_list0,
        next_pos_list1,
    )
    total_num = len(next_pos_list0)
    dumbbell_list0, dumbbell_list1 = [], []
    for x, y, next_pos0, next_pos1 in progressbar(
        iterator,
        total=total_num,
        desc="Finding dumbbells",
        disable=not show_progressbar,
    ):
        mask1 = _make_circular_mask(
            next_pos0[1], next_pos0[0], s.data.shape[0], s.data.shape[1], mask_radius
        )
        mask2 = _make_circular_mask(
            next_pos1[1], next_pos1[0], s.data.shape[0], s.data.shape[1], mask_radius
        )
        pos1_sum = (s.data * mask1).sum()
        pos2_sum = (s.data * mask2).sum()
        if pos1_sum > pos2_sum:
            dumbbell_list0.append([x, y])
            dumbbell_list1.append(next_pos0)
        else:
            dumbbell_list0.append(next_pos1)
            dumbbell_list1.append([x, y])
    dumbbell_list0 = np.array(dumbbell_list0)
    dumbbell_list1 = np.array(dumbbell_list1)
    return (dumbbell_list0, dumbbell_list1)


def make_atom_lattice_dumbbell_structure(
    s, dumbbell_positions, dumbbell_vector, show_progressbar=True
):
    """Make Atom_Lattice object from image of dumbbell structure.

    Parameters
    ----------
    s : HyperSpy 2D signal
    dumbbell_positions : list of atomic positions
        In the form [[x0, y0], [x1, y1], [x2, y2], ...].
        There should only be one position per dumbbell.
    dumbbell_vector : tuple
    show_progressbar : bool, default True

    Returns
    -------
    dumbbell_lattice: Atomap Dumbbell_Lattice object

    Examples
    --------
    >>> import atomap.initial_position_finding as ipf
    >>> s = am.dummy_data.get_dumbbell_signal()
    >>> dumbbell_positions = am.get_atom_positions(s, separation=16)
    >>> atom_positions = am.get_atom_positions(s, separation=4)
    >>> dumbbell_vector = ipf.find_dumbbell_vector(atom_positions)
    >>> dumbbell_lattice = ipf.make_atom_lattice_dumbbell_structure(
    ...     s, dumbbell_positions, dumbbell_vector)

    """
    dumbbell_list0, dumbbell_list1 = _get_dumbbell_arrays(
        s, dumbbell_positions, dumbbell_vector, show_progressbar=show_progressbar
    )
    s_modified = do_pca_on_signal(s)
    sublattice0 = Sublattice(
        atom_position_list=dumbbell_list0,
        original_image=s.data,
        image=s_modified.data,
        color="blue",
    )
    sublattice1 = Sublattice(
        atom_position_list=dumbbell_list1,
        original_image=s.data,
        image=s_modified.data,
        color="red",
    )
    sublattice0.find_nearest_neighbors()
    sublattice1.find_nearest_neighbors()
    atom_lattice = Dumbbell_Lattice(
        image=sublattice0.image,
        original_image=s.data,
        name="Dumbbell structure",
        sublattice_list=[sublattice0, sublattice1],
    )
    return atom_lattice


class AtomAdderRemover:
    def __init__(self, image, atom_positions=None, distance_threshold=4, norm="linear"):
        self.image = image
        self.distance_threshold = distance_threshold
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Use the left mouse button to add or remove atoms")
        if norm == "linear":
            self.cax = self.ax.imshow(self.image)
        elif norm == "log":
            value_min = np.min(self.image.__array__())
            log_norm = LogNorm(1e-20, vmax=np.max(self.image.__array__()) - value_min)
            self.cax = self.ax.imshow(self.image.__array__(), norm=log_norm)
        if atom_positions is None:
            self.atom_positions = []
        else:
            if hasattr(atom_positions, "tolist"):
                atom_positions = atom_positions.tolist()
            self.atom_positions = copy.deepcopy(atom_positions)
        x_pos, y_pos = self.get_xy_pos_lists()
        (self.line,) = self.ax.plot(x_pos, y_pos, "o", color="red")
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.fig.tight_layout()

    def onclick(self, event):
        if event.inaxes != self.ax.axes:
            return
        if event.button == 1:  # Left mouse button
            x = np.float(event.xdata)
            y = np.float(event.ydata)
            atom_nearby = self.is_atom_nearby(x, y)
            if atom_nearby is False:
                self.atom_positions.append([x, y])
            else:
                self.atom_positions.pop(atom_nearby)
            self.replot()

    def is_atom_nearby(self, x_press, y_press):
        dt = self.distance_threshold
        index = False
        closest_dist = 9999999999999999
        x_pos, y_pos = self.get_xy_pos_lists()
        for i, (x, y) in enumerate(zip(x_pos, y_pos)):
            if x - dt < x_press < x + dt:
                if y - dt < y_press < y + dt:
                    dist = math.hypot(x_press - x, y_press - y)
                    if dist < closest_dist:
                        index = i
        return index

    def get_xy_pos_lists(self):
        if self.atom_positions:
            x_pos_list = np.array(self.atom_positions)[:, 0]
            y_pos_list = np.array(self.atom_positions)[:, 1]
        else:
            x_pos_list = []
            y_pos_list = []
        return (x_pos_list, y_pos_list)

    def replot(self):
        x_pos, y_pos = self.get_xy_pos_lists()
        self.line.set_xdata(x_pos)
        self.line.set_ydata(y_pos)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def add_atoms_with_gui(image, atom_positions=None, distance_threshold=4, norm="linear"):
    """Add or remove atoms from a list of atom positions.

    Will open a matplotlib figure, where atoms can be added or
    removed by pressing them.

    Parameters
    ----------
    image : array-like
        Signal or NumPy array
    atom_positions : list of lists, optional
        In the form [[x0, y0], [x1, y1], ...]
    distance_threshold : int
        Default 4
    norm : string, optional
        'linear' or 'log', default 'linear'. If set up log,
        the intensity will be visualized as a log plot.
        Useful to see low-intensity atoms.

    Returns
    -------
    atom_positions : list of lists
        In the form [[x0, y0], [x1, y1], ...].
        The list can be updated until the figure is closed.

    Examples
    --------
    >>> s = am.dummy_data.get_simple_cubic_signal()
    >>> atom_positions = am.get_atom_positions(s, separation=9)
    >>> atom_positions_new = am.add_atoms_with_gui(atom_positions, s)

    Log norm

    >>> atom_positions_new = am.add_atoms_with_gui(
    ...     atom_positions, s, norm='log')

    """
    global atom_adder_remover
    atom_adder_remover = AtomAdderRemover(
        image, atom_positions, distance_threshold=distance_threshold, norm=norm
    )
    atom_positions_new = atom_adder_remover.atom_positions
    return atom_positions_new


def select_atoms_with_gui(image, atom_positions, verts=None, invert_selection=False):
    """Get a subset of a list of atom positions.

    Will open a matplotlib figure, where a polygon is drawn interactively.

    Parameters
    ----------
    image : array-like
        Signal or NumPy array
    atom_positions : list of lists, or NumPy array
        In the form [[x0, y0], [x1, y1], ...]
    verts : list of tuples
        List of positions, spanning an enclosed region.
        [(x0, y0), (x1, y1), ...]. Need to have at least 3 positions.
    invert_selection : bool, optional
        Get the atom positions outside the region, instead of the
        ones inside it. Default False.

    Returns
    -------
    atom_positions_selected : list of lists

    Examples
    --------
    >>> s = am.dummy_data.get_simple_cubic_signal()
    >>> atom_positions = am.get_atom_positions(s, separation=9)
    >>> atom_positions_selected = am.select_atoms_with_gui(
    ...        s, atom_positions)
    >>> sublattice = am.Sublattice(atom_positions_selected, s)

    Using the function non-interactively

    >>> verts = [(60, 235), (252, 236), (147, 58)]
    >>> atom_positions_selected = am.select_atoms_with_gui(
    ...        s, atom_positions, verts=verts)
    >>> sublattice = am.Sublattice(atom_positions_selected, s)

    Getting the atom positions outside the selected regions, instead of
    outside it

    >>> atom_positions_selected = am.select_atoms_with_gui(
    ...        s, atom_positions, verts=verts, invert_selection=True)
    >>> sublattice = am.Sublattice(atom_positions_selected, s)

    """
    if verts is None:
        global atom_selector
        atom_selector = gc.GetAtomSelection(
            image, atom_positions, invert_selection=invert_selection
        )
        atom_positions_selected = atom_selector.atom_positions_selected
    else:
        atom_positions_selected = to._get_atom_selection_from_verts(
            atom_positions, verts, invert_selection=invert_selection
        )
    return atom_positions_selected
