import pytest
from pytest import approx
import numpy as np
import atomap.api as am
import atomap.tools as to
import atomap.atom_position as ap
from hyperspy.signals import Signal2D
from atomap.tools import array2signal1d, array2signal2d, Fingerprinter
from atomap.tools import remove_atoms_from_image_using_2d_gaussian
import atomap.dummy_data as dd
from atomap.tools import integrate, calculate_point_record
from atomap.tools import find_smallest_distance, remove_integrated_edge_cells
from atomap.tools import _border_elems
import atomap.testing_tools as tt
import hyperspy as hs


class TestArray2Signal:
    def test_array2signal2d(self):
        array = np.arange(100).reshape(10, 10)
        scale = 0.2
        units = "nm"
        s = array2signal2d(array, scale=scale, units=units)
        assert s.axes_manager[0].scale == scale
        assert s.axes_manager[1].scale == scale
        assert s.axes_manager[0].units == units
        assert s.axes_manager[1].units == units
        assert s.axes_manager.shape == array.shape

    def test_array2signal1d(self):
        array = np.arange(100)
        scale = 0.2
        s = array2signal1d(array, scale=scale)
        assert s.axes_manager[0].scale == scale
        assert s.axes_manager.shape == array.shape


class TestGetPointBetweenFourAtoms:
    def test_simple(self):
        atom0, atom1 = ap.Atom_Position(10, 30), ap.Atom_Position(20, 30)
        atom2, atom3 = ap.Atom_Position(10, 40), ap.Atom_Position(20, 40)
        mid_pos = to.get_point_between_four_atoms((atom0, atom1, atom2, atom3))
        assert mid_pos == (15.0, 35.0)

    def test_negative(self):
        atom0, atom1 = ap.Atom_Position(-15, 30), ap.Atom_Position(-10, 30)
        atom2, atom3 = ap.Atom_Position(-15, 40), ap.Atom_Position(-10, 40)
        mid_pos = to.get_point_between_four_atoms((atom0, atom1, atom2, atom3))
        assert mid_pos == (-12.5, 35.0)

    def test_diagonal(self):
        atom0, atom1 = ap.Atom_Position(5, 0), ap.Atom_Position(0, 5)
        atom2, atom3 = ap.Atom_Position(5, 10), ap.Atom_Position(10, 5)
        mid_pos = to.get_point_between_four_atoms((atom0, atom1, atom2, atom3))
        assert mid_pos == (5.0, 5.0)

    def test_wrong_size(self):
        atom0, atom1 = ap.Atom_Position(5, 0), ap.Atom_Position(0, 5)
        atom2, atom3 = ap.Atom_Position(5, 10), ap.Atom_Position(10, 5)
        atom4 = ap.Atom_Position(10, 10)
        with pytest.raises(ValueError):
            to.get_point_between_four_atoms((atom0, atom1, atom2, atom3, atom4))


class TestFingerprinter:
    def test_running_simple(self):
        xN, yN, n = 3, 3, 60
        point_list = tt.make_nn_test_dataset(xN=xN, yN=yN, n=n)
        fp = Fingerprinter()
        fp.fit(point_list)
        clusters = (xN * 2 + 1) * (yN * 2 + 1) - 1
        assert fp.cluster_centers_.shape[0] == clusters

    def test_running_advanced(self):
        xN, yN, xS, yS, n = 2, 2, 9, 12, 70
        tolerance = 0.5
        point_list = tt.make_nn_test_dataset(xN=xN, yN=yN, xS=xS, yS=yS, n=n)
        fp = Fingerprinter()
        fp.fit(point_list)
        bool_list = []
        for ix in list(range(-xN, xN + 1)):
            for iy in list(range(-yN, yN + 1)):
                if (ix == 0) and (iy == 0):
                    pass
                else:
                    for point in point_list:
                        x, y = ix * xS, iy * yS
                        xD, yD = abs(x - point[0]), abs(y - point[1])
                        if (xD < tolerance) and (yD < tolerance):
                            bool_list.append(True)
                            break
        clusters = (xN * 2 + 1) * (yN * 2 + 1) - 1
        assert len(bool_list) == clusters


class TestAddZeroPositionToDataList:
    def test_simple(self):
        x_array, y_array = np.arange(10), np.arange(10, 20)
        z_array = np.arange(30, 40)
        x_array_extra, y_array_extra = np.arange(5), np.arange(5)
        x_new, y_new, z_new = to._add_zero_position_to_data_list(
            x_array, y_array, z_array, x_array_extra, y_array_extra
        )
        assert len(x_array) + len(x_array_extra) == len(x_new)
        assert len(y_array) + len(y_array_extra) == len(y_new)
        assert len(z_array) + len(x_array_extra) == len(z_new)
        assert len(x_new) == len(y_new) == len(z_new)


class TestRemoveAtomsFromImageUsing2dGaussian:
    def setup_method(self):
        test_data = tt.MakeTestData(520, 520)
        x, y = np.mgrid[10:510:8j, 10:510:8j]
        x, y = x.flatten(), y.flatten()
        test_data.add_atom_list(x, y)
        sublattice = test_data.sublattice
        self.sublattice = sublattice

    def test_running(self):
        sublattice = self.sublattice
        sublattice.find_nearest_neighbors()
        remove_atoms_from_image_using_2d_gaussian(
            sublattice.image, sublattice, percent_to_nn=0.40
        )

    def test_no_nearest_neighbors(self):
        sublattice = self.sublattice
        with pytest.raises(ValueError):
            remove_atoms_from_image_using_2d_gaussian(sublattice.image, sublattice)


class TestFindAverageDistanceBetweenAtoms:
    def test_simple(self):
        position_list = []
        for i in range(0, 100, 9):
            positions = np.ones(10) * i
            position_list.extend(positions)

        position_list = np.array(position_list)
        property_list = np.ones(len(position_list))
        input_data_list = np.stack((position_list, property_list), axis=1)

        output = to.find_average_distance_between_atoms(input_data_list)
        first_peak, monolayer_sep, mean_separation = output

        assert first_peak == 9.0
        assert len(monolayer_sep) == 11
        assert (monolayer_sep.flatten() == 9.0).all()
        assert mean_separation == 9.0

    def test_crop(self):
        position_list = []
        for i in range(0, 100, 9):
            positions = np.ones(10) * i
            position_list.extend(positions)

        position_list = np.array(position_list)
        property_list = np.ones(len(position_list))
        input_data_list = np.stack((position_list, property_list), axis=1)

        output = to.find_average_distance_between_atoms(
            input_data_list, crop_start=10, crop_end=10
        )
        first_peak, monolayer_sep, mean_separation = output

        assert first_peak == 9.0
        assert len(monolayer_sep) == 9
        assert (monolayer_sep.flatten() == 9.0).all()
        assert mean_separation == 9.0

    def test_with_random(self):
        position_list = []
        for i in range(0, 100, 9):
            positions = np.random.random(size=1000) + i
            position_list.extend(positions)

        position_list = np.array(position_list)
        property_list = np.ones(len(position_list))
        input_data_list = np.stack((position_list, property_list), axis=1)

        output = to.find_average_distance_between_atoms(input_data_list)
        first_peak, monolayer_sep, mean_separation = output

        assert approx(first_peak, abs=0.1) == 8.0
        assert len(monolayer_sep) == 11
        assert monolayer_sep == approx(8, abs=0.1)
        assert mean_separation == approx(8, abs=0.1)


class TestSortPositionsIntoLayers:
    def test_simple(self):
        position_list = []
        for i in range(0, 100, 9):
            positions = np.ones(10) * i
            position_list.extend(positions)

        position_list = np.array(position_list)
        property_list = np.ones(len(position_list))
        data_list = np.stack((position_list, property_list), axis=1)
        layer_list = to.sort_positions_into_layers(data_list, 4.5)
        assert len(layer_list) == 12
        for layer, i in zip(layer_list, range(0, 100, 9)):
            assert len(layer) == 10
            assert np.array(layer)[:, 0] == approx(i)
            assert np.array(layer)[:, 1] == approx(1.0)


class TestSortProjectedPositionsIntoLayers:
    def test_simple(self):
        position_list = []
        for i in range(0, 100, 9):
            positions = np.ones(10) * i
            position_list.extend(positions)

        position_list = np.array(position_list)
        property_list = np.ones(len(position_list))
        data_list = np.stack((position_list, property_list), axis=1)
        layer_list = to.sort_projected_positions_into_layers(data_list)
        assert len(layer_list) == 12
        for layer, i in zip(layer_list, range(0, 100, 9)):
            assert len(layer) == 10
            assert np.array(layer)[:, 0] == approx(i)
            assert np.array(layer)[:, 1] == approx(1.0)


class TestCombineProjectedPositionsLayers:
    def test_simple(self):
        position_list = []
        for i in range(0, 100, 9):
            positions = np.ones(10) * i
            position_list.extend(positions)

        position_list = np.array(position_list)
        property_list = np.ones(len(position_list))
        data_list = np.stack((position_list, property_list), axis=1)
        layer_list = to.sort_projected_positions_into_layers(data_list)
        combined_layer_list = to.combine_projected_positions_layers(layer_list)
        assert len(layer_list) == 12
        for layer, i in zip(combined_layer_list, range(0, 100, 9)):
            assert layer[0] == i
            assert layer[1] == 1.0
            assert layer[2] == 0.0

    def test_std(self):
        n = 10
        position_list = []
        property_list = []
        for i in range(0, 100, 9):
            positions = np.ones(n) * i
            position_list.extend(positions)
            property_list.extend(np.ones(int(n / 2)))
            property_list.extend(np.ones(int(n / 2)) * 2)

        data_list = np.stack((position_list, property_list), axis=1)
        layer_list = to.sort_projected_positions_into_layers(data_list)
        combined_layer_list = to.combine_projected_positions_layers(layer_list)
        for layer, i in zip(combined_layer_list, range(0, 100, 9)):
            assert layer[0] == i
            assert layer[1] == 1.5
            assert layer[2] == 0.5


@pytest.mark.parametrize(
    "x,y,sx,sy,ox,oy",
    [
        (20, 20, 1, 1, 0, 0),
        (50, 20, 1, 1, 0, 0),
        (20, 20, 0.5, 2, 0, 0),
        (20, 20, 0.5, 2, 10, 5),
        (50, 20, 0.5, 4, -10, 20),
        (15, 50, 0.7, 2.3, -10, -5),
    ],
)
def test_get_signal_centre(x, y, sx, sy, ox, oy):
    s = Signal2D(np.zeros((y, x)))
    print(s)
    am = s.axes_manager
    am[0].scale, am[0].offset, am[1].scale, am[1].offset = sx, ox, sy, oy
    xC, yC = to._get_signal_centre(s)
    print(xC, yC)
    assert xC == ((0.5 * (x - 1) * sx) + ox)
    assert yC == ((0.5 * (y - 1) * sy) + oy)


class TestRotatePointsAroundSignalCentre:
    def setup_method(self):
        sublattice = dd.get_simple_cubic_sublattice()
        self.x, self.y = sublattice.x_position, sublattice.y_position
        self.s = sublattice.get_atom_list_on_image()

    @pytest.mark.parametrize("rot", [10, 30, 60, 90, 180, 250])
    def test_simple_rotation(self, rot):
        x_rot, y_rot = to.rotate_points_around_signal_centre(
            self.s, self.x, self.y, rot
        )
        assert len(self.x) == len(x_rot)
        assert len(self.y) == len(y_rot)

    @pytest.mark.parametrize("rot", [0, 360])
    def test_zero_rotation(self, rot):
        x_rot, y_rot = to.rotate_points_around_signal_centre(
            self.s, self.x, self.y, rot
        )
        np.testing.assert_allclose(self.x, x_rot)
        np.testing.assert_allclose(self.y, y_rot)

    def test_180_double_rotation(self):
        x_rot, y_rot = to.rotate_points_around_signal_centre(
            self.s, self.x, self.y, 180
        )
        x_rot, y_rot = to.rotate_points_around_signal_centre(self.s, x_rot, y_rot, 180)
        np.testing.assert_allclose(self.x, x_rot)
        np.testing.assert_allclose(self.y, y_rot)

    def test_90_double_rotation(self):
        x_rot, y_rot = to.rotate_points_around_signal_centre(self.s, self.x, self.y, 90)
        x_rot, y_rot = to.rotate_points_around_signal_centre(self.s, x_rot, y_rot, -90)
        np.testing.assert_allclose(self.x, x_rot)
        np.testing.assert_allclose(self.y, y_rot)


class TestRotatePointsAndSignal:
    def setup_method(self):
        sublattice = dd.get_simple_cubic_sublattice()
        self.x, self.y = sublattice.x_position, sublattice.y_position
        self.s = sublattice.get_atom_list_on_image()

    def test_simple_rotation(self):
        s_rot, x_rot, y_rot = to.rotate_points_and_signal(self.s, self.x, self.y, 180)
        assert self.s.axes_manager.shape == s_rot.axes_manager.shape
        assert len(self.x) == len(x_rot)
        assert len(self.y) == len(y_rot)

    def test_double_180_rotation(self):
        s_rot, x_rot, y_rot = to.rotate_points_and_signal(self.s, self.x, self.y, 180)
        s_rot, x_rot, y_rot = to.rotate_points_and_signal(s_rot, x_rot, y_rot, 180)
        np.testing.assert_allclose(self.x, x_rot)
        np.testing.assert_allclose(self.y, y_rot)


class TestRotatePointsAroundPosition:
    def test_degrees_centre_0(self):
        xc, yc = 0, 0
        x, y = [5,], [
            0,
        ]
        x_rot, y_rot = to._rotate_points_around_position(xc, yc, x, y, 90)
        assert len(x_rot)
        assert len(y_rot)
        assert approx(x_rot[0]) == 0
        assert approx(y_rot[0]) == -5
        x_rot, y_rot = to._rotate_points_around_position(xc, yc, x, y, -90)
        assert approx(x_rot[0]) == 0
        assert approx(y_rot[0]) == 5
        x_rot, y_rot = to._rotate_points_around_position(xc, yc, x, y, 180)
        assert approx(x_rot[0]) == -5
        assert approx(y_rot[0]) == 0
        x_rot, y_rot = to._rotate_points_around_position(xc, yc, x, y, -180)
        assert approx(x_rot[0]) == -5
        assert approx(y_rot[0]) == 0

    def test_degrees_centre_10(self):
        xc, yc = 10, 10
        x, y = [5,], [
            0,
        ]
        x_rot, y_rot = to._rotate_points_around_position(xc, yc, x, y, 90)
        assert approx(x_rot[0]) == 0
        assert approx(y_rot[0]) == 15
        x_rot, y_rot = to._rotate_points_around_position(xc, yc, x, y, -90)
        assert approx(x_rot[0]) == 20
        assert approx(y_rot[0]) == 5
        x_rot, y_rot = to._rotate_points_around_position(xc, yc, x, y, 180)
        assert approx(x_rot[0]) == 15
        assert approx(y_rot[0]) == 20
        x_rot, y_rot = to._rotate_points_around_position(xc, yc, x, y, -180)
        assert approx(x_rot[0]) == 15
        assert approx(y_rot[0]) == 20

    def test_many_positions(self):
        x, y = np.arange(100, 300), np.arange(400, 600)
        x_rot, y_rot = to._rotate_points_around_position(5, 5, x, y, 90)
        assert len(x) == len(x_rot)
        assert len(y) == len(y_rot)


class TestFliplrPointsAroundSignalCentre:
    def setup_method(self):
        sublattice = dd.get_simple_cubic_sublattice()
        self.x, self.y = sublattice.x_position, sublattice.y_position
        self.s = sublattice.get_atom_list_on_image()

    def test_simple_flip(self):
        x_flip, y_flip = to.fliplr_points_around_signal_centre(self.s, self.x, self.y)
        assert len(self.x) == len(x_flip)
        assert len(self.y) == len(y_flip)

    def test_flip_double(self):
        x_flip, y_flip = to.fliplr_points_around_signal_centre(self.s, self.x, self.y)
        x_flip, y_flip = to.fliplr_points_around_signal_centre(self.s, x_flip, y_flip)
        np.testing.assert_allclose(self.x, x_flip)
        np.testing.assert_allclose(self.y, y_flip)


class TestFliplrPointsAndSignal:
    def setup_method(self):
        sublattice = dd.get_simple_cubic_sublattice()
        self.x, self.y = sublattice.x_position, sublattice.y_position
        self.s = sublattice.get_atom_list_on_image()

    def test_simple_flip(self):
        s_flip, x_flip, y_flip = to.fliplr_points_and_signal(self.s, self.x, self.y)
        s_flip, x_flip, y_flip = to.fliplr_points_and_signal(s_flip, x_flip, y_flip)
        np.testing.assert_allclose(self.s.data, s_flip.data)
        np.testing.assert_allclose(self.x, x_flip)
        np.testing.assert_allclose(self.y, y_flip)


class TestIntegrate:
    def test_two_atoms(self):
        test_data = tt.MakeTestData(50, 100, sigma_quantile=8)
        x, y, A = [25, 25], [25, 75], [5, 10]
        test_data.add_atom_list(x=x, y=y, amplitude=A)
        s = test_data.signal
        i_points, i_record, p_record = integrate(s, x, y, max_radius=500)

        assert approx(i_points) == A
        assert i_record.axes_manager.signal_shape == (50, 100)
        assert (i_record.isig[:, :51].data == i_points[0]).all()
        assert (i_record.isig[:, 51:].data == i_points[1]).all()
        assert (p_record.data[:, 51:] == 0).all()
        assert (p_record.data[:, 51:] == 1).all()

    def test_four_atoms(self):
        test_data = tt.MakeTestData(60, 100, sigma_quantile=8)
        x, y, A = [20, 20, 40, 40], [25, 75, 25, 75], [5, 10, 15, 20]
        test_data.add_atom_list(x=x, y=y, amplitude=A)
        s = test_data.signal
        i_points, i_record, p_record = integrate(s, x, y, max_radius=500)

        assert approx(i_points) == A
        assert (i_record.isig[:31, :51].data == i_points[0]).all()
        assert (i_record.isig[:31, 51:].data == i_points[1]).all()
        assert (i_record.isig[31:, :51].data == i_points[2]).all()
        assert (i_record.isig[31:, 51:].data == i_points[3]).all()
        assert (p_record.data[:51, :31] == 0).all()
        assert (p_record.data[51:, :31] == 1).all()
        assert (p_record.data[:51, 31:] == 2).all()
        assert (p_record.data[51:, 31:] == 3).all()

    def test_max_radius_bad_value(self):
        s = hs.signals.Signal2D(np.zeros((10, 10)))
        with pytest.raises(ValueError):
            integrate(
                s,
                [
                    5,
                ],
                [
                    5,
                ],
                max_radius=-1,
            )

    def test_max_radius_1(self):
        test_data = tt.MakeTestData(60, 100)
        x, y, A = [30, 30], [25, 75], [5, 10]
        test_data.add_atom_list(x=x, y=y, amplitude=A, sigma_x=0.1, sigma_y=0.1)
        s = test_data.signal
        i_points, i_record, p_record = integrate(s, x, y, max_radius=1)

        assert (i_points[1] / i_points[0]) == 2.0
        assert i_record.data[y[0], x[0]] == i_points[0]
        assert i_record.data[y[1], x[1]] == i_points[1]
        i_record.data[y[0], x[0]] = 0
        i_record.data[y[1], x[1]] = 0
        i_record.data[np.isnan(i_record.data)] = 0
        assert not i_record.data.any()

    def test_too_few_dimensions(self):
        s = hs.signals.Signal1D(np.random.rand(110))
        y, x = np.mgrid[5:96:10, 5:96:10]
        x, y = x.flatten(), y.flatten()
        with pytest.raises(ValueError):
            integrate(s, x, y)

    def test_sum_2d_random_data(self):
        s = hs.signals.Signal2D(np.random.rand(100, 110))
        y, x = np.mgrid[5:96:10, 5:96:10]
        x, y = x.flatten(), y.flatten()
        result = integrate(s, x, y)
        assert approx(np.sum(result[0])) == np.sum(s.data)
        assert result[2].data.shape == s.data.shape

    def test_3d_data_running(self):
        s = dd.get_eels_spectrum_survey_image()
        s_eels = dd.get_eels_spectrum_map().T
        peaks = am.get_atom_positions(s, separation=4)
        i_points, i_record, p_record = integrate(
            s_eels, peaks[:, 0], peaks[:, 1], max_radius=3
        )
        assert p_record.data.shape == (100, 100)
        assert s_eels.data.shape == i_record.data.shape

    def test_watershed_method_running(self):
        test_data = tt.MakeTestData(60, 100)
        x, y, A = [20, 20, 40, 40], [25, 75, 25, 75], [5, 10, 15, 20]
        test_data.add_atom_list(x=x, y=y, amplitude=A)
        s = test_data.signal
        i_points, i_record, p_record = integrate(s, x, y, method="Watershed")

    def test_wrong_method(self):
        s = hs.signals.Signal2D(np.zeros((10, 10)))
        with pytest.raises(NotImplementedError):
            integrate(
                s,
                [
                    5,
                ],
                [
                    5,
                ],
                method="bad_method",
            )

    def test_array_input(self):
        sublattice = am.dummy_data.get_simple_cubic_sublattice()
        x, y = sublattice.x_position, sublattice.y_position
        i_points0, i_record0, p_record0 = integrate(sublattice.image, x, y)

        signal = am.dummy_data.get_simple_cubic_signal()
        i_points1, i_record1, p_record1 = integrate(signal, x, y)
        assert (i_points0 == i_points1).all()
        assert (i_record0.data == i_record1.data).all()
        assert (p_record0.data == p_record1.data).all()

    def test_find_smallest_distance(self):
        points = np.array([(2, 2), (1, 1), (0, 0), (3, 3)]).T
        minIndex, distMin = find_smallest_distance(6, 6, points)
        assert minIndex == 3
        assert distMin == ((6 - 3) ** 2 + (6 - 3) ** 2) ** 0.5

    def test_calculate_point_record(self):
        points = np.array([(2, 2), (1, 1), (0, 0), (3, 3)]).T
        image = np.zeros((3, 3))
        max_radius = 2
        point_record = calculate_point_record(image, points, max_radius)
        np.testing.assert_almost_equal(
            point_record, np.array([[3.0, 2.0, 2.0], [2.0, 2.0, 1.0], [2.0, 1.0, 1.0]])
        )

    def test_border_elems(self):
        a = np.arange(5)[:, None] * np.arange(6)
        np.testing.assert_almost_equal(
            _border_elems(a, 1),
            np.array([0, 0, 0, 0, 0, 0, 0, 5, 0, 10, 0, 15, 0, 4, 8, 12, 16, 20]),
        )


class TestVoronoi:
    def setup_method(self):
        np.random.seed(1)
        s = am.dummy_data.get_dumbbell_heterostructure_signal()
        self.s = s.isig[:64, :64]
        self.points = am.get_atom_positions(self.s, separation=3).T

        self.i_true = np.array(
            [
                47.04196098,
                np.nan,
                np.nan,
                52.47312043,
                np.nan,
                np.nan,
                np.nan,
                52.85817517,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                42.53379219,
                np.nan,
                38.09332358,
                np.nan,
                37.73917033,
            ]
        )

    def test_integrate_voronoi(self):
        i, ir, pr = am.integrate(
            self.s,
            *self.points,
            method="Voronoi",
            remove_edge_cells=True,
            edge_pixels=3,
        )
        np.testing.assert_almost_equal(i, self.i_true)

    def test_remove_edges(self):
        i, ir, pr = am.integrate(
            self.s,
            *self.points,
            method="Voronoi",
            remove_edge_cells=False,
        )
        i, ir, pr = remove_integrated_edge_cells(i, ir, pr, 3)
        np.testing.assert_almost_equal(i, self.i_true)


class TestGetAtomSelectionFromVerts:
    def test_simple(self):
        pos = [[10, 10], [15, 15]]
        verts = [[0, 0], [0, 20], [20, 20], [20, 0]]
        pos_selected = to._get_atom_selection_from_verts(pos, verts)
        assert len(pos_selected) == 2
        assert (pos_selected == [[10, 10], [15, 15]]).all()

    def test_numpy_input(self):
        pos = np.array([[10, 10], [15, 15]])
        verts = [[0, 0], [0, 20], [20, 20], [20, 0]]
        pos_selected = to._get_atom_selection_from_verts(pos, verts)
        assert len(pos_selected) == 2
        assert (pos_selected == [[10, 10], [15, 15]]).all()

    def test_inside_outside(self):
        pos = [[10, 10], [25, 10]]
        verts = [[0, 0], [0, 20], [20, 20], [20, 0]]
        pos_selected = to._get_atom_selection_from_verts(pos, verts)
        assert len(pos_selected) == 1
        assert (pos_selected == [10, 10]).all()

    def test_all_outside(self):
        pos = [[10, 25], [25, 10]]
        verts = [[0, 0], [0, 20], [20, 20], [20, 0]]
        pos_selected = to._get_atom_selection_from_verts(pos, verts)
        assert len(pos_selected) == 0

    def test_too_few_verts(self):
        pos = [[10, 25], [25, 10]]
        verts = [[0, 0], [0, 20]]
        with pytest.raises(ValueError):
            to._get_atom_selection_from_verts(pos, verts)

    def test_many_positions(self):
        pos = np.random.randint(10, 30, size=(1000, 2))
        verts = [[15, 15], [15, 25], [25, 25], [25, 15]]
        pos_selected = to._get_atom_selection_from_verts(pos, verts)
        pos_x, pos_y = pos_selected[:, 0], pos_selected[:, 1]
        assert len(pos_x[pos_x < 15]) == 0
        assert len(pos_x[pos_x > 25]) == 0
        assert len(pos_y[pos_y < 15]) == 0
        assert len(pos_y[pos_y > 25]) == 0
