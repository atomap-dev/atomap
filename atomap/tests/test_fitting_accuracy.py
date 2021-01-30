from pytest import approx
import numpy as np
from atomap.testing_tools import MakeTestData
from atomap.testing_tools import find_atom_position_match
from atomap.testing_tools import get_fit_miss_array
import atomap.api as am


class TestFittingAccuracy:
    def setup_method(self):
        test_data = MakeTestData(700, 700)
        x, y = np.mgrid[100:600:5j, 100:600:5j]
        sigma_value = 10
        test_data.add_atom_list(
            x.flatten(),
            y.flatten(),
            sigma_x=sigma_value,
            sigma_y=sigma_value,
            amplitude=50,
        )
        self.s = test_data.signal
        self.g_list = test_data.gaussian_list
        self.sigma_value = sigma_value

    def test_center_of_mass(self):
        g_list = self.g_list
        s = self.s
        atom_lattice = am.make_atom_lattice_from_image(
            s, am.process_parameters.GenericStructure(), pixel_separation=90
        )
        sublattice = atom_lattice.sublattice_list[0]
        sublattice.refine_atom_positions_using_center_of_mass(sublattice.original_image)
        sublattice.refine_atom_positions_using_center_of_mass(sublattice.original_image)
        atom_list = sublattice.atom_list
        match_list = find_atom_position_match(
            g_list, atom_list, scale=sublattice.pixel_size, delta=3
        )
        fit_miss = get_fit_miss_array(match_list)
        mean_diff = fit_miss[:, 2].mean()
        assert approx(mean_diff, abs=1e-4) == 0.0

    def test_gaussian_2d(self):
        g_list = self.g_list
        s = self.s
        atom_lattice = am.make_atom_lattice_from_image(
            s, am.process_parameters.GenericStructure(), pixel_separation=90
        )
        sublattice = atom_lattice.sublattice_list[0]
        atom_list = sublattice.atom_list
        match_list = find_atom_position_match(
            g_list, atom_list, scale=sublattice.pixel_size, delta=3
        )
        fit_miss = get_fit_miss_array(match_list)
        mean_diff = fit_miss[:, 2].mean()
        assert approx(mean_diff, abs=1e-7) == 0.0
        sigma_x_list = []
        sigma_y_list = []
        for atom in atom_list:
            sigma_x_list.append(atom.sigma_x)
            sigma_y_list.append(atom.sigma_y)
        mean_sigma_x = np.array(sigma_x_list).mean()
        mean_sigma_y = np.array(sigma_y_list).mean()
        assert approx(mean_sigma_x, rel=1e-3) == self.sigma_value
        assert approx(mean_sigma_y, rel=1e-3) == self.sigma_value
