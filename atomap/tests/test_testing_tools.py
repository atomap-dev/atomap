import unittest
import pytest
from pytest import approx
import numpy as np
import atomap.testing_tools as tt


class TestMakeTestData:
    def test_simple_init(self):
        tt.MakeTestData(100, 100)

    def test_show_progressbar(self):
        test_data0 = tt.MakeTestData(100, 100, show_progressbar=True)
        assert test_data0._show_progressbar

        test_data1 = tt.MakeTestData(100, 100, show_progressbar=False)
        assert not test_data1._show_progressbar

    def test_get_signal(self):
        imX0, imY0 = 100, 100
        test_data0 = tt.MakeTestData(imX0, imY0)
        assert (imX0, imY0) == test_data0.data_extent
        assert test_data0.signal.axes_manager.shape == (imX0, imY0)
        assert not test_data0.signal.data.any()

        imX1, imY1 = 100, 39
        test_data1 = tt.MakeTestData(imX1, imY1)
        assert (imX1, imY1) == test_data1.data_extent
        assert test_data1.signal.axes_manager.shape == (imX1, imY1)

        imX2, imY2 = 34, 65
        test_data2 = tt.MakeTestData(imX2, imY2)
        assert (imX2, imY2) == test_data2.data_extent
        assert test_data2.signal.axes_manager.shape == (imX2, imY2)

    def test_add_image_noise(self):
        test_data0 = tt.MakeTestData(1000, 1000)
        mu0, sigma0 = 0, 0.005
        test_data0.add_image_noise(mu=mu0, sigma=sigma0, only_positive=False)
        s0 = test_data0.signal
        assert approx(s0.data.mean(), abs=1e-4) == mu0
        assert approx(s0.data.std(), abs=1e-2) == sigma0

        test_data1 = tt.MakeTestData(1000, 1000)
        mu1, sigma1 = 10, 0.5
        test_data1.add_image_noise(mu=mu1, sigma=sigma1, only_positive=False)
        s1 = test_data1.signal
        assert approx(s1.data.mean(), rel=1e-3) == mu1
        assert approx(s1.data.std(), abs=1e-2) == sigma1

        test_data2 = tt.MakeTestData(1000, 1000)
        mu2, sigma2 = 154.2, 1.98
        test_data2.add_image_noise(mu=mu2, sigma=sigma2, only_positive=False)
        s2 = test_data2.signal
        assert approx(s2.data.mean(), rel=1e-3) == mu2
        assert approx(s2.data.std(), rel=1e-2) == sigma2

    def test_add_image_noise_only_positive(self):
        test_data0 = tt.MakeTestData(1000, 1000)
        test_data0.add_image_noise(mu=0, sigma=0.005, only_positive=True)
        s0 = test_data0.signal
        assert (s0.data > 0).all()

    def test_add_image_noise_random_seed(self):
        test_data0 = tt.MakeTestData(100, 100)
        test_data0.add_image_noise(random_seed=0)
        s0 = test_data0.signal
        test_data1 = tt.MakeTestData(100, 100)
        test_data1.add_image_noise(random_seed=0)
        s1 = test_data1.signal
        assert (s0.data == s1.data).all()

    def test_add_atom(self):
        x, y, sx, sy, A, r = 10, 5, 5, 9, 10, 2
        td = tt.MakeTestData(50, 50)
        td.add_atom(x, y, sigma_x=sx, sigma_y=sy, amplitude=A, rotation=r)
        assert len(td.sublattice.atom_list) == 1
        atom = td.sublattice.atom_list[0]
        assert atom.pixel_x == x
        assert atom.pixel_y == y
        assert atom.sigma_x == sx
        assert atom.sigma_y == sy
        assert atom.amplitude_gaussian == A
        assert atom.rotation == r

    def test_add_atom_list_simple(self):
        x, y = np.mgrid[10:90:10, 10:90:10]
        x, y = x.flatten(), y.flatten()
        sx, sy, A, r = 2.1, 1.3, 9.5, 1.4
        td = tt.MakeTestData(100, 100)
        td.add_atom_list(x=x, y=y, sigma_x=sx, sigma_y=sy, amplitude=A, rotation=r)
        atom_list = td.sublattice.atom_list
        assert len(atom_list) == len(x)
        for tx, ty, atom in zip(x, y, atom_list):
            assert atom.pixel_x == tx
            assert atom.pixel_y == ty
            assert atom.sigma_x == sx
            assert atom.sigma_y == sy
            assert atom.amplitude_gaussian == A
            assert atom.rotation == r

    def test_add_atom_list_all_lists(self):
        x, y = np.mgrid[10:90:10, 10:90:10]
        x, y = x.flatten(), y.flatten()
        sx = np.random.random_sample(size=len(x))
        sy = np.random.random_sample(size=len(x))
        A = np.random.random_sample(size=len(x))
        r = np.random.random_sample(size=len(x))
        td = tt.MakeTestData(100, 100)
        td.add_atom_list(x=x, y=y, sigma_x=sx, sigma_y=sy, amplitude=A, rotation=r)
        atom_list = td.sublattice.atom_list
        assert len(atom_list) == len(x)

        iterator = zip(x, y, sx, sy, A, r, atom_list)
        for tx, ty, tsx, tsy, tA, tr, atom in iterator:
            assert atom.pixel_x == tx
            assert atom.pixel_y == ty
            assert atom.sigma_x == tsx
            assert atom.sigma_y == tsy
            assert atom.amplitude_gaussian == tA
            assert atom.rotation == tr

    def test_add_atom_list_wrong_input(self):
        x, y = np.mgrid[10:90:10, 10:90:10]
        x, y = x.flatten(), y.flatten()
        td = tt.MakeTestData(100, 100)
        with pytest.raises(ValueError):
            td.add_atom_list(x, y[10:])

        sx = np.arange(10)
        with pytest.raises(ValueError):
            td.add_atom_list(x, y, sigma_x=sx)

        sy = np.arange(20)
        with pytest.raises(ValueError):
            td.add_atom_list(x, y, sigma_y=sy)

        A = np.arange(30)
        with pytest.raises(ValueError):
            td.add_atom_list(x, y, amplitude=A)

        r = np.arange(5)
        with pytest.raises(ValueError):
            td.add_atom_list(x, y, rotation=r)

    def test_gaussian_list(self):
        x, y = np.mgrid[10:90:10, 10:90:10]
        x, y = x.flatten(), y.flatten()
        sx = np.random.random_sample(size=len(x))
        sy = np.random.random_sample(size=len(x))
        A = np.random.random_sample(size=len(x))
        r = np.random.random_sample(size=len(x))
        td = tt.MakeTestData(100, 100)
        td.add_atom_list(x=x, y=y, sigma_x=sx, sigma_y=sy, amplitude=A, rotation=r)
        gaussian_list = td.gaussian_list
        assert len(gaussian_list) == len(x)

        iterator = zip(x, y, sx, sy, A, r, gaussian_list)
        for tx, ty, tsx, tsy, tA, tr, gaussian in iterator:
            assert gaussian.centre_x.value == tx
            assert gaussian.centre_y.value == ty
            assert gaussian.sigma_x.value == tsx
            assert gaussian.sigma_y.value == tsy
            assert gaussian.A.value == tA
            assert gaussian.rotation.value == tr

    def test_sublattice_generate_image(self):
        testdata = tt.MakeTestData(1000, 1000, sublattice_generate_image=False)
        x, y = np.mgrid[0:1000:150j, 0:1000:150j]
        x, y = x.flatten(), y.flatten()
        testdata.add_atom_list(x, y)
        sublattice = testdata.sublattice
        assert (sublattice.image == 0).all()
        assert len(sublattice.atom_list) == 150 * 150

    def test_sigma_quantile(self):
        testdata0 = tt.MakeTestData(500, 500, sigma_quantile=5)
        testdata1 = tt.MakeTestData(500, 500, sigma_quantile=1)
        x, y = np.mgrid[0:490:30, 0:490:30]
        x, y = x.flatten(), y.flatten()
        testdata0.add_atom_list(x, y)
        testdata1.add_atom_list(x, y)
        assert testdata0.signal.data.sum() > testdata1.signal.data.sum()

    def test_add_row_scan_distortion(self):
        x, y = np.mgrid[10:90:10, 10:90:10]
        x, y = x.flatten(), y.flatten()
        td = tt.MakeTestData(100, 100)
        td.add_atom_list(x=x, y=y, sigma_x=2, sigma_y=2)
        s0 = td.signal
        td = tt.MakeTestData(100, 100, add_row_scan_distortion=1)
        td.add_atom_list(x=x, y=y, sigma_x=2, sigma_y=2)
        s1 = td.signal
        assert not np.all(s0.data == s1.data)


class TestMakeVectorTestGaussian(unittest.TestCase):
    def test_running(self):
        x, y, std, n = 10, 5, 0.5, 5000
        point_list = tt.make_vector_test_gaussian(x, y, standard_deviation=std, n=n)
        point_list_meanX = point_list[:, 0].mean()
        point_list_meanY = point_list[:, 1].mean()
        point_list_stdX = point_list[:, 0].std()
        point_list_stdY = point_list[:, 1].std()

        assert approx(point_list_meanX, rel=1e-2) == x
        assert approx(point_list_meanY, rel=1e-2) == y
        assert approx(point_list_stdX, rel=1e-1) == std
        assert approx(point_list_stdY, rel=1e-1) == std
        assert n == point_list.shape[0]


class TestMakeNnTestDataset(unittest.TestCase):
    def test_running(self):
        xN, yN, n = 4, 4, 60
        point_list = tt.make_nn_test_dataset(xN=xN, yN=yN, n=n)

        total_point = n * (((2 * xN) + 1) * ((2 * yN) + 1) - 1)
        assert point_list.shape[0] == total_point
