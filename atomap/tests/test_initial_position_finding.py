from pytest import approx
import numpy as np
import atomap.api as am
from atomap.testing_tools import MakeTestData
import atomap.initial_position_finding as ipf


class TestFindDumbbellVector:
    def test_5_separation_x(self):
        test_data = MakeTestData(200, 200)
        x0, y0 = np.mgrid[10:200:20, 10:200:20]
        x1, y1 = np.mgrid[16:200:20, 10:200:20]
        x, y = np.vstack((x0, x1)).flatten(), np.vstack((y0, y1)).flatten()
        test_data.add_atom_list(x, y, sigma_x=1, sigma_y=1, amplitude=50)
        atom_positions = am.get_atom_positions(test_data.signal, 4)
        vector = ipf.find_dumbbell_vector(atom_positions)
        assert approx(abs(vector[0]), abs=1e-7) == 6.0
        assert approx(abs(vector[1]), abs=1e-7) == 0.0

    def test_5_separation_y(self):
        test_data = MakeTestData(200, 200)
        x0, y0 = np.mgrid[10:200:20, 10:200:20]
        x1, y1 = np.mgrid[10:200:20, 16:200:20]
        x, y = np.vstack((x0, x1)).flatten(), np.vstack((y0, y1)).flatten()
        test_data.add_atom_list(x, y, sigma_x=1, sigma_y=1, amplitude=50)
        atom_positions = am.get_atom_positions(test_data.signal, 4)
        vector = ipf.find_dumbbell_vector(atom_positions)
        assert approx(abs(vector[0]), abs=1e-7) == 0.0
        assert approx(abs(vector[1]), abs=1e-7) == 6.0

    def test_3x_3y_separation(self):
        test_data = MakeTestData(200, 200)
        x0, y0 = np.mgrid[10:200:20, 10:200:20]
        x1, y1 = np.mgrid[13:200:20, 13:200:20]
        x, y = np.vstack((x0, x1)).flatten(), np.vstack((y0, y1)).flatten()
        test_data.add_atom_list(x, y, sigma_x=1, sigma_y=1, amplitude=50)
        atom_positions = am.get_atom_positions(test_data.signal, 3)
        vector = ipf.find_dumbbell_vector(atom_positions)
        assert approx(abs(vector[0]), abs=1e-7) == 3.0
        assert approx(abs(vector[1]), abs=1e-7) == 3.0


class TestGetDumbbellArrays:
    def setup_method(self):
        test_data = MakeTestData(230, 230)
        x0, y0 = np.mgrid[20:210:20, 20:210:20]
        x1, y1 = np.mgrid[26:210:20, 20:210:20]
        x, y = np.vstack((x0, x1)).flatten(), np.vstack((y0, y1)).flatten()
        test_data.add_atom_list(x, y, sigma_x=1, sigma_y=1, amplitude=50)
        self.s = test_data.signal

    def test_simple_running(self):
        s = self.s
        atom_positions = am.get_atom_positions(s, separation=4)
        vector = ipf.find_dumbbell_vector(atom_positions)
        dumbbell_positions = am.get_atom_positions(s, 14)
        dumbbell_array0, dumbbell_array1 = ipf._get_dumbbell_arrays(
            s, dumbbell_positions, vector
        )
        assert len(dumbbell_array0) == 100
        assert len(dumbbell_array1) == 100


class TestMakeAtomLatticeDumbbellStructure:
    def setup_method(self):
        test_data = MakeTestData(230, 230)
        x0, y0 = np.mgrid[20:210:20, 20:210:20]
        x1, y1 = np.mgrid[26:210:20, 20:210:20]
        x, y = np.vstack((x0, x1)).flatten(), np.vstack((y0, y1)).flatten()
        test_data.add_atom_list(x, y, sigma_x=1, sigma_y=1, amplitude=50)
        self.s = test_data.signal

    def test_simple_running(self):
        s = self.s
        atom_positions = am.get_atom_positions(s, separation=4)
        vector = ipf.find_dumbbell_vector(atom_positions)
        dumbbell_positions = am.get_atom_positions(s, separation=13)
        atom_lattice = ipf.make_atom_lattice_dumbbell_structure(
            s, dumbbell_positions, vector
        )
        assert len(atom_lattice.sublattice_list) == 2
        sublattice0 = atom_lattice.sublattice_list[0]
        sublattice1 = atom_lattice.sublattice_list[1]
        assert len(sublattice0.atom_list) == 100
        assert len(sublattice1.atom_list) == 100
