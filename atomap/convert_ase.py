from scipy.ndimage.filters import gaussian_filter
from hyperspy.misc.elements import elements
from atomap import atom_lattice, sublattice
import numpy as np
import hyperspy.api as hs


def load_ase(atoms, image_size=(1024, 1024), gaussian_blur=3):
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
    >>> from atomap.convert_ase import load_ase
    >>> atoms = Octahedron('Ag', 5, cutoff=2)
    >>> atomlattice = load_ase(atoms)

    """

    columns = {}
    for atom in atoms:
        if (atom.x, atom.y) in columns:
            columns[(atom.x, atom.y)][0].append(atom.z)
            columns[(atom.x, atom.y)][1].append(atom.symbol)
        else:
            columns[(atom.x, atom.y)] = [[atom.z], [atom.symbol]]

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
            sublattice_dict[composition_str]['xy'].append(list(xy))
            sublattice_dict[composition_str]['el_info'].append(column)
        else:
            sublattice_dict[composition_str] = {}
            sublattice_dict[composition_str]['xy'] = [list(xy)]
            sublattice_dict[composition_str]['el_info'] = [column]

    image_array, axes_dict = _generate_image_from_ase(atoms,
                                                      image_size,
                                                      gaussian_blur)
    image = hs.signals.Signal2D(image_array)

    sublattice_colors = ['green', 'blue', 'red']
    sublattice_list = []
    i = -1
    for composition, sublattice_items in sublattice_dict.items():
        xy = np.asarray(sublattice_items['xy'])
        xy[:, 0] = xy[:, 0]/axes_dict[0]['scale']
        xy[:, 1] = xy[:, 1]/axes_dict[1]['scale']
        sublattice_list.append(sublattice.Sublattice(xy,
                                                     image,
                                                     pixel_size=axes_dict[0][
                                                         'scale']/10,
                                                     color=sublattice_colors[
                                                         i]))
        i -= 1

    for lattice in sublattice_list:
        for atom in lattice.atom_list:
            atom.set_element_info(columns[(atom.pixel_x*axes_dict[0]['scale'],
                                           atom.pixel_y*axes_dict[1]['scale'])][1],
                                  columns[(atom.pixel_x*axes_dict[0]['scale'],
                                           atom.pixel_y*axes_dict[1]['scale'])][0])

    atomlattice = atom_lattice.Atom_Lattice(image=image,
                                            sublattice_list=sublattice_list)

    return(atomlattice)


def _generate_image_from_ase(
        atoms,
        image_size=(1024, 1024),
        gaussian_blur=3):
    image_array = np.zeros(image_size)

    offset_axis0 = atoms.positions[:, 0].min()
    offset_axis1 = atoms.positions[:, 1].min()

    if offset_axis0 == 0.0 or offset_axis1 == 0.0:
        offset_axis = atoms.positions[:, 0].max()/10

    scale_axis0 = (atoms.positions[:, 0].max() + offset_axis)/image_size[0]
    scale_axis1 = (atoms.positions[:, 1].max() + offset_axis)/image_size[1]

    for atom in atoms:
        atom_Z = elements[atom.symbol]['General_properties']['Z']

        index_axis0 = int(round(atom.x/scale_axis0))
        index_axis1 = int(round(atom.y/scale_axis1))

        image_array[index_axis0, index_axis1] += atom_Z

    gaussian_filter(image_array, gaussian_blur, output=image_array)

    axisx_dict = {
            'scale': scale_axis0,
            'offset': offset_axis0}
    axisy_dict = {
            'scale': scale_axis1,
            'offset': offset_axis1}

    axes_dict = [axisx_dict, axisy_dict]
    return(image_array, axes_dict)
