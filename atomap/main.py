from atomap.atom_finding_refining import (
    subtract_average_background,
    do_pca_on_signal,
    refine_sublattice,
    construct_zone_axes_from_sublattice,
    get_atom_positions,
    normalize_signal,
)

from atomap.tools import remove_atoms_from_image_using_2d_gaussian

from atomap.atom_lattice import Atom_Lattice
from atomap.sublattice import Sublattice
import atomap.process_parameters as pp


def run_image_filtering(signal, invert_signal=False):
    """Subtracts background, filters noise with PCA, and normalizes a signal.

    Parameters
    ----------
    signal : HyperSpy Signal2D
    invert_signal : bool, default False
        Inverts the image to 1./signal.data

    Returns
    -------
    filtered_signal : HyperSpy Signal2D

    Example
    -------
    >>> import atomap.api as am
    >>> s = am.dummy_data.get_simple_cubic_signal()
    >>> from atomap.main import run_image_filtering
    >>> s_new = run_image_filtering(s)

    """
    signal.change_dtype("float64")
    signal_modified = subtract_average_background(signal)
    signal_modified = do_pca_on_signal(signal_modified)
    signal_modified = normalize_signal(signal_modified, invert_signal=invert_signal)
    if invert_signal:
        signal.data = 1.0 / signal.data
    return signal_modified


def _get_signal_name(signal):
    filename = None
    signal_dict = signal.__dict__
    if "metadata" in signal_dict:
        if "General" in signal_dict["metadata"]:
            if "title" in signal_dict["metadata"]["General"]:
                temp_title = signal_dict["metadata"]["General"]["title"]
                if not temp_title == "":
                    filename = temp_title
    if filename is None:
        if "tmp_parameters" in signal_dict:
            if "filename" in signal_dict["tmp_parameters"]:
                temp_filename = signal_dict["tmp_parameters"]["filename"]
                if not temp_filename == "":
                    filename = temp_filename
    if filename is None:
        filename = "signal"
    return filename


def make_atom_lattice_from_image(
    s_image0,
    process_parameter=None,
    pixel_separation=None,
    s_image1=None,
    debug_plot=False,
):
    if s_image0.data.dtype == "float16":
        raise ValueError(
            "s_image0 has the dtype float16, which is not supported. "
            "Convert it to something else, for example using "
            "s_image0.change_dtype('float64')"
        )
    image0_filename = _get_signal_name(s_image0)

    name = image0_filename

    s_image0 = s_image0.deepcopy()
    s_image0_modified = run_image_filtering(s_image0)

    if process_parameter is None:
        process_parameter = pp.GenericStructure()

    image0_scale = s_image0.axes_manager[0].scale
    if pixel_separation is None:
        if process_parameter.peak_separation is None:
            raise ValueError(
                "pixel_separation is not set.\
                    Either set it in the process_parameter.peak_separation\
                    or pixel_separation parameter"
            )
        else:
            pixel_separation = process_parameter.peak_separation / image0_scale
    initial_atom_position_list = get_atom_positions(
        s_image0_modified, separation=pixel_separation
    )

    if s_image1 is not None:
        if s_image1.data.dtype == "float16":
            raise ValueError(
                "s_image1 has the dtype float16, which is not supported. "
                "Convert it to something else, for example using "
                "s_image1.change_dtype('float64')"
            )
        s_image1 = s_image1.deepcopy()
        s_image1.data = 1.0 / s_image1.data
        image1_data = s_image1.data

    #################################

    image0_data = s_image0.data
    image0_data_modified = s_image0_modified.data

    atom_lattice = Atom_Lattice(name=name)
    atom_lattice._original_filename = image0_filename
    atom_lattice.image = image0_data
    if s_image1 is not None:
        atom_lattice.image_extra = image1_data
    atom_lattice._pixel_separation = pixel_separation

    for sublattice_index in range(process_parameter.number_of_sublattices):
        sublattice_para = process_parameter.get_sublattice_from_order(sublattice_index)

        if sublattice_para.image_type == 0:
            s_image = s_image0
            image_data = image0_data
            image_data_modified = image0_data_modified
        if sublattice_para.image_type == 1:
            if s_image1 is not None:
                s_image = s_image1
                image_data = image1_data
                image_data_modified = image1_data
            else:
                break

        if sublattice_para.sublattice_order == 0:
            sublattice = Sublattice(initial_atom_position_list, image_data_modified)
        else:
            temp_sublattice = atom_lattice.get_sublattice(
                sublattice_para.sublattice_position_sublattice
            )
            temp_zone_vector_index = temp_sublattice.get_zone_vector_index(
                sublattice_para.sublattice_position_zoneaxis
            )
            zone_vector = temp_sublattice.zones_axis_average_distances[
                temp_zone_vector_index
            ]
            atom_list = temp_sublattice.find_missing_atoms_from_zone_vector(zone_vector)

            sublattice = Sublattice(atom_list, image_data)

        zone_axis_para_list = False
        if hasattr(sublattice_para, "zone_axis_list"):
            zone_axis_para_list = sublattice_para.zone_axis_list

        sublattice._plot_color = sublattice_para.color
        sublattice.name = sublattice_para.name
        sublattice.pixel_size = s_image.axes_manager[0].scale
        sublattice._pixel_separation = pixel_separation
        sublattice.original_image = image_data
        atom_lattice.sublattice_list.append(sublattice)
        if debug_plot:
            sublattice.plot_atom_list_on_image_data(
                figname=sublattice.name + "_initial_position.jpg"
            )
        for atom in sublattice.atom_list:
            atom.sigma_x = sublattice._pixel_separation / 10.0
            atom.sigma_y = sublattice._pixel_separation / 10.0
        if not (sublattice_para.sublattice_order == 0):
            construct_zone_axes_from_sublattice(
                sublattice, zone_axis_para_list=zone_axis_para_list
            )
            atom_subtract_config = sublattice_para.atom_subtract_config
            image_data = sublattice.image
            for atom_subtract_para in atom_subtract_config:
                temp_sublattice = atom_lattice.get_sublattice(
                    atom_subtract_para["sublattice"]
                )
                neighbor_distance = atom_subtract_para["neighbor_distance"]
                image_data = remove_atoms_from_image_using_2d_gaussian(
                    image_data, temp_sublattice, percent_to_nn=neighbor_distance
                )
            sublattice.image = image_data
            sublattice.original_image = image_data

        refinement_config = sublattice_para.refinement_config
        refinement_neighbor_distance = refinement_config["neighbor_distance"]
        refinement_steps = refinement_config["config"]
        for refinement_step in refinement_steps:
            if refinement_step[0] == "image_data":
                refinement_step[0] = sublattice.original_image
            elif refinement_step[0] == "image_data_modified":
                refinement_step[0] = sublattice.image
            else:
                refinement_step[0] = sublattice.original_image

        refine_sublattice(sublattice, refinement_steps, refinement_neighbor_distance)

        if sublattice_para.sublattice_order == 0:
            sublattice.construct_zone_axes(zone_axis_para_list=zone_axis_para_list)

    return atom_lattice
