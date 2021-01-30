from scipy.ndimage.filters import gaussian_filter
from hyperspy.misc.elements import elements
from atomap import atom_lattice, sublattice
import numpy as np
import hyperspy.api as hs


def ase_to_atom_lattice(atoms, image_size=None, gaussian_blur=3):
    """
    Load Atom_lattice object from an ASE Atoms object.

    Parameters
    ----------
    atoms : ASE Atoms object
    image_size : tuple
    gaussian_blur : int

    Returns
    -------
    atomlattice : Atom_lattice object

    Examples
    --------
    >>> from ase.cluster import Octahedron
    >>> from atomap.convert_ase import ase_to_atom_lattice
    >>> atoms = Octahedron('Ag', 10, cutoff=2)
    >>> atomlattice = ase_to_atom_lattice(atoms)

    """
    image_array, axes_dict = _generate_image_from_ase(atoms, image_size, gaussian_blur)
    image = hs.signals.Signal2D(image_array)

    scale_x, scale_y = axes_dict[0]["scale"], axes_dict[1]["scale"]
    offset_x, offset_y = axes_dict[0]["offset"], axes_dict[1]["offset"]

    columns = {}
    for atom in atoms:
        ax = (atom.x - offset_x) / scale_x
        ay = (atom.y - offset_y) / scale_y
        if (ax, ay) in columns:
            columns[(ax, ay)][0].append(atom.z)
            columns[(ax, ay)][1].append(atom.symbol)
        else:
            columns[(ax, ay)] = [[atom.z], [atom.symbol]]

    sublattice_dict = {}
    for xy, column in columns.items():
        sum_el = {}
        for el in column[1]:
            if el in sum_el:
                sum_el[el] += 1
            else:
                sum_el[el] = 1

        composition = {}
        for el in sum_el:
            composition[el] = sum_el[el] / sum(sum_el.values())
        composition_str = str(composition)

        if composition_str in sublattice_dict:
            sublattice_dict[composition_str]["xy"].append(list(xy))
            sublattice_dict[composition_str]["el_info"].append(column)
        else:
            sublattice_dict[composition_str] = {}
            sublattice_dict[composition_str]["xy"] = [list(xy)]
            sublattice_dict[composition_str]["el_info"] = [column]

    sublattice_colors = ["green", "blue", "red"]
    sublattice_list = []
    i = -1
    for composition, sublattice_items in sublattice_dict.items():
        xy = np.asarray(sublattice_items["xy"])
        sublattice_list.append(
            sublattice.Sublattice(
                xy,
                image,
                pixel_size=scale_x / 10,
                color=sublattice_colors[i],
            )
        )
        i -= 1

    for lattice in sublattice_list:
        for atom in lattice.atom_list:
            atom_key = (atom.pixel_x, atom.pixel_y)
            atom.set_element_info(columns[atom_key][1], columns[atom_key][0])
    atomlattice = atom_lattice.Atom_Lattice(
        image=image, sublattice_list=sublattice_list
    )

    return atomlattice


def _generate_image_from_ase(atoms, image_size=None, gaussian_blur=3):
    min_axis_x = atoms.positions[:, 0].min()
    min_axis_y = atoms.positions[:, 1].min()
    max_axis_x = atoms.positions[:, 0].max()
    max_axis_y = atoms.positions[:, 1].max()

    max_size = min(max_axis_x - min_axis_x, max_axis_y - min_axis_y)
    padding = max_size * 0.1

    min_axis_x -= padding
    min_axis_y -= padding
    max_axis_x += padding
    max_axis_y += padding
    size_axis_x = max_axis_x - min_axis_x
    size_axis_y = max_axis_y - min_axis_y

    if image_size is None:
        image_max_size = 1024
        if size_axis_x >= size_axis_y:
            image_size_x = int(image_max_size)
            image_size_y = int(image_max_size * size_axis_y / size_axis_x)
        else:
            image_size_x = int(image_max_size * size_axis_x / size_axis_y)
            image_size_y = int(image_max_size)
    else:
        image_size_y, image_size_x = image_size

    image_array = np.zeros((image_size_y, image_size_x))

    offset_axis_x, offset_axis_y = min_axis_x, min_axis_y
    scale_axis_x = size_axis_x / image_size_x
    scale_axis_y = size_axis_y / image_size_y
    scale_max = max(scale_axis_x, scale_axis_y)

    for atom in atoms:
        atom_Z = elements[atom.symbol]["General_properties"]["Z"]

        index_axis_x = int(round((atom.x - offset_axis_x) / scale_max))
        index_axis_y = int(round((atom.y - offset_axis_y) / scale_max))

        image_array[index_axis_y, index_axis_x] += atom_Z

    gaussian_filter(image_array, gaussian_blur, output=image_array)

    axisx_dict = {"scale": scale_max, "offset": offset_axis_x}
    axisy_dict = {"scale": scale_max, "offset": offset_axis_y}

    axes_dict = [axisx_dict, axisy_dict]
    return (image_array, axes_dict)
