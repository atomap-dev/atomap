import os
import numpy as np
from hyperspy.api import load
import atomap.atom_finding_refining as afr
from atomap.sublattice import Sublattice
from atomap.atom_lattice import Atom_Lattice

my_path = os.path.dirname(__file__)
s_adf_filename = os.path.join(my_path, "datasets", "test_ADF_cropped.hdf5")
s_abf_filename = os.path.join(my_path, "datasets", "test_ABF_cropped.hdf5")


class TestAdfAbfStoAutoprocess:
    def setup_method(self):
        peak_separation = 0.15

        s_adf = load(s_adf_filename)
        s_adf.change_dtype("float64")
        s_adf_modified = afr.subtract_average_background(s_adf)
        self.s_adf_modified = afr.do_pca_on_signal(s_adf_modified)
        self.pixel_size = s_adf.axes_manager[0].scale
        self.pixel_separation = peak_separation / self.pixel_size

        s_abf = load(s_abf_filename)
        s_abf.change_dtype("float64")

        self.peaks = afr.get_atom_positions(self.s_adf_modified, self.pixel_separation)

    def test_find_b_cation_atoms(self):
        a_sublattice = Sublattice(
            self.peaks, np.rot90(np.fliplr(self.s_adf_modified.data))
        )
        a_sublattice.pixel_size = self.pixel_size
        afr.construct_zone_axes_from_sublattice(a_sublattice)

        zone_vector_100 = a_sublattice.zones_axis_average_distances[1]
        b_atom_list = a_sublattice.find_missing_atoms_from_zone_vector(zone_vector_100)
        b_sublattice = Sublattice(
            b_atom_list, np.rot90(np.fliplr(self.s_adf_modified.data))
        )
        assert len(b_sublattice.atom_list) == 104

    def test_find_b_atom_planes(self):
        a_sublattice = Sublattice(
            self.peaks, np.rot90(np.fliplr(self.s_adf_modified.data))
        )
        a_sublattice.pixel_size = self.pixel_size
        afr.construct_zone_axes_from_sublattice(a_sublattice)

        zone_vector_100 = a_sublattice.zones_axis_average_distances[1]
        b_atom_list = a_sublattice.find_missing_atoms_from_zone_vector(zone_vector_100)
        b_sublattice = Sublattice(
            b_atom_list, np.rot90(np.fliplr(self.s_adf_modified.data))
        )
        b_sublattice.pixel_size = self.pixel_size
        afr.construct_zone_axes_from_sublattice(b_sublattice)


class TestAdfAbfStoManualprocess:
    def test_manual_processing(self):
        s_adf_main = load(s_adf_filename)
        s = s_adf_main.deepcopy()
        s.change_dtype("float32")
        atom_positions = afr.get_atom_positions(
            signal=s,
            separation=17,
            threshold_rel=0.02,
        )
        sublattice = Sublattice(atom_position_list=atom_positions, image=s.data)
        sublattice.find_nearest_neighbors()
        sublattice.refine_atom_positions_using_center_of_mass(
            sublattice.image, percent_to_nn=0.4
        )
        sublattice.refine_atom_positions_using_2d_gaussian(
            sublattice.image, percent_to_nn=0.4
        )

        Atom_Lattice(image=s.data, sublattice_list=[sublattice])
        sublattice.construct_zone_axes()
