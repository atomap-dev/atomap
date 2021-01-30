import pytest
import numpy as np
from numpy.testing import assert_allclose
from hyperspy.signals import Signal1D, Signal2D
from atomap.atom_position import Atom_Position
from atomap.sublattice import Sublattice
from atomap.testing_tools import MakeTestData
import atomap.dummy_data as dd
import atomap.atom_finding_refining as afr


class TestMakeMaskFromPositions:
    def test_radius_1(self):
        x, y, r = 10, 20, 1
        pos = [[x, y]]
        rad = [r]
        mask = afr._make_mask_from_positions(pos, rad, (40, 40))
        assert mask.sum() == 5.0
        mask[x, y] = False
        mask[x + r, y] = False
        mask[x - r, y] = False
        mask[x, y + 1] = False
        mask[x, y - 1] = False
        assert not mask.any()

    def test_2_positions_radius_1(self):
        x0, y0, x1, y1, r = 10, 20, 20, 30, 1
        pos = [[x0, y0], [x1, y1]]
        rad = [r, r]
        mask = afr._make_mask_from_positions(pos, rad, (40, 40))
        assert mask.sum() == 10.0
        mask[x0, y0] = False
        mask[x0 + r, y0] = False
        mask[x0 - r, y0] = False
        mask[x0, y0 + 1] = False
        mask[x0, y0 - 1] = False
        mask[x1, y1] = False
        mask[x1 + r, y1] = False
        mask[x1 - r, y1] = False
        mask[x1, y1 + 1] = False
        mask[x1, y1 - 1] = False
        assert not mask.any()

    def test_radius_2(self):
        x, y, r = 10, 5, 2
        pos = [[x, y]]
        rad = [r]
        mask = afr._make_mask_from_positions(pos, rad, (40, 40))
        assert mask.sum() == 13.0

    def test_2_positions_radius_2(self):
        x0, y0, x1, y1, r = 5, 7, 17, 25, 2
        pos = [[x0, y0], [x1, y1]]
        rad = [r, r]
        mask = afr._make_mask_from_positions(pos, rad, (40, 40))
        assert mask.sum() == 26.0

    def test_wrong_input(self):
        x, y, r = 10, 5, 2
        pos = [[x, y]]
        rad = [r, r]
        with pytest.raises(ValueError):
            afr._make_mask_from_positions(
                position_list=pos, radius_list=rad, data_shape=(40, 40)
            )


class TestRemoveTooCloseAtoms:
    def test_simple(self):
        afr._remove_too_close_atoms(np.array([[0, 1]]), 5)

    def test_two_atoms(self):
        data = np.array([[1, 10], [10, 1]])
        data_new0 = afr._remove_too_close_atoms(data, 5)
        assert (data_new0 == data).all()
        data_new1 = afr._remove_too_close_atoms(data, 20)
        assert len(data_new1) == 1
        assert (data_new1 == data[0]).all()

    def test_three_atoms(self):
        data = np.array([[1, 10], [10, 1], [3, 10]])
        data_new = afr._remove_too_close_atoms(data, 5)
        assert (data_new == data[:2]).all()

    def test_several_overlap(self):
        data = np.array([[1, 10], [10, 1], [10, 10]])
        data_new0 = afr._remove_too_close_atoms(data, 5)
        assert (data_new0 == data).all()
        data_new1 = afr._remove_too_close_atoms(data, 20)
        assert len(data_new1) == 1
        assert (data_new1 == data[0]).all()

    def test_many_atoms(self):
        x, y = np.meshgrid(np.arange(10, 90, 10), np.arange(10, 90, 10))
        x, y = x.flatten(), y.flatten()
        data = np.stack((x, y)).T
        data_new0 = afr._remove_too_close_atoms(data, 5)
        assert (data_new0 == data).all()
        data_new1 = afr._remove_too_close_atoms(data, 20)
        assert len(data_new1) != len(data_new0)
        data_new2 = afr._remove_too_close_atoms(data, 200)
        assert len(data_new2)
        assert (data_new2 == data[0]).all()


class TestCropMask:
    def test_radius_1(self):
        x, y, r = 10, 20, 1
        pos = [[x, y]]
        rad = [r]
        mask = afr._make_mask_from_positions(pos, rad, (40, 40))
        x0, x1, y0, y1 = afr._crop_mask_slice_indices(mask)
        assert x0 == x - r
        assert x1 == x + r + 1
        assert y0 == y - r
        assert y1 == y + r + 1
        mask_crop = mask[x0:x1, y0:y1]
        assert mask_crop.shape == (2 * r + 1, 2 * r + 1)

    def test_radius_2(self):
        x, y, r = 15, 10, 2
        pos = [[x, y]]
        rad = [r]
        mask = afr._make_mask_from_positions(pos, rad, (40, 40))
        x0, x1, y0, y1 = afr._crop_mask_slice_indices(mask)
        mask_crop = mask[x0:x1, y0:y1]
        assert mask_crop.shape == (2 * r + 1, 2 * r + 1)

    def test_radius_5(self):
        x, y, r = 15, 10, 5
        pos = [[x, y]]
        rad = [r]
        mask = afr._make_mask_from_positions(pos, rad, (40, 40))
        x0, x1, y0, y1 = afr._crop_mask_slice_indices(mask)
        mask_crop = mask[x0:x1, y0:y1]
        assert mask_crop.shape == (2 * r + 1, 2 * r + 1)


class TestCropAndPadArray:
    def test_crop(self):
        arr = np.array([[100]])
        arr2 = np.zeros((9, 9))
        arr2[4, 4] = 100
        assert np.all(afr._crop_array(arr, 0, 0, 5) == arr2)
        assert not np.all(afr._crop_array(arr, 1, 0, 5) == arr2)
        assert not np.all(afr._crop_array(arr, 0, 1, 5) == arr2)
        assert afr._crop_array(arr, 0, 0, 4).shape != arr2.shape

    @pytest.mark.parametrize(
        "dtype", ["uint8", "uint16", "uint32", "float16", "float32", "bool"]
    )
    def test_dtypes(self, dtype):
        arr = np.ones((20, 30), dtype=dtype)
        data = afr._crop_array(arr, 10, 15, 5)
        # Should return float64 dtype
        assert data.dtype == np.float64

    def test_pad_array(self):
        arr = np.ones((2, 2))
        arr2 = afr._pad_array(arr, 1)
        assert arr2.sum() == arr.sum()
        assert arr2.shape == (4, 4)


class TestFindBackgroundValue:
    def test_percentile(self):
        data = np.arange(100)
        value = afr._find_background_value(data, lowest_percentile=0.01)
        assert value == 0.0
        value = afr._find_background_value(data, lowest_percentile=0.1)
        assert value == 4.5
        value = afr._find_background_value(data, lowest_percentile=0.5)
        assert value == 24.5


class TestFindMedianUpperPercentile:
    def test_percentile(self):
        data = np.arange(100)
        value = afr._find_median_upper_percentile(data, upper_percentile=0.01)
        assert value == 99.0
        value = afr._find_median_upper_percentile(data, upper_percentile=0.1)
        assert value == 94.5
        value = afr._find_median_upper_percentile(data, upper_percentile=0.5)
        assert value == 74.5


class TestMakeModelFromAtomList:
    def setup_method(self):
        image_data = np.random.random(size=(100, 100))
        position_list = []
        for x in range(10, 100, 5):
            for y in range(10, 100, 5):
                position_list.append([x, y])
        sublattice = Sublattice(np.array(position_list), image_data)
        sublattice.find_nearest_neighbors()
        self.sublattice = sublattice

    def test_1_atom(self):
        sublattice = self.sublattice
        model, mask = afr._make_model_from_atom_list(
            [sublattice.atom_list[10]], sublattice.image
        )
        assert len(model) == 1

    def test_2_atom(self):
        sublattice = self.sublattice
        model, mask = afr._make_model_from_atom_list(
            sublattice.atom_list[10:12], sublattice.image
        )
        assert len(model) == 2

    def test_5_atom(self):
        sublattice = self.sublattice
        model, mask = afr._make_model_from_atom_list(
            sublattice.atom_list[10:15], sublattice.image
        )
        assert len(model) == 5

    def test_set_mask_radius_atom(self):
        atom_list = [Atom_Position(2, 2), Atom_Position(4, 4)]
        image = np.random.random((20, 20))
        model, mask = afr._make_model_from_atom_list(
            atom_list=atom_list, image_data=image, mask_radius=3
        )
        assert len(model) == 2


class TestCenterOfMass:
    def test_find_center(self):
        center = np.zeros((5, 5))
        center[1, 1] = 1
        assert tuple(afr.calculate_center_of_mass(center)) == (1, 1)

    def test_compare_center_of_mass(self):
        from scipy.ndimage import center_of_mass

        rand = np.random.random((5, 5))
        com0 = center_of_mass(rand)
        com1 = tuple(afr.calculate_center_of_mass(rand))
        assert pytest.approx(com0, rel=1e-12) == com1

    def test_non_square0(self):
        image = np.zeros((40, 20))
        y, x = 38, 11
        image[y, x] = 10.0
        cy, cx = afr.calculate_center_of_mass(image)
        assert cy == float(y)
        assert cx == float(x)

    def test_non_square1(self):
        image = np.zeros((40, 80))
        y, x = 38, 11
        image[y, x] = 10.0
        cy, cx = afr.calculate_center_of_mass(image)
        assert cy == float(y)
        assert cx == float(x)

    def test_nd_2D_array(self):
        image = np.zeros((3, 4, 40, 80))
        y, x = 38, 11
        image[:, :, y, x] = 10.0
        com = afr.calculate_center_of_mass(image)
        assert com.shape == (3, 4, 2)
        cy, cx = com.T
        assert (cy == float(y)).all()
        assert (cx == float(x)).all()

    def test_nd_2D_array_positions(self):
        data = np.zeros((2, 3, 20, 10), dtype=np.uint16)
        data[0, 0, 12, 5] = 2
        data[0, 1, 9, 8] = 2
        data[0, 2, 1, 1] = 2
        data[1, 0, 3, 5] = 2
        data[1, 1, 19, 2] = 2
        data[1, 2, 11, 6] = 2
        com = afr.calculate_center_of_mass(data)
        assert list(com[0, 0]) == [12.0, 5.0]
        assert list(com[0, 1]) == [9.0, 8.0]
        assert list(com[0, 2]) == [1.0, 1.0]
        assert list(com[1, 0]) == [3.0, 5.0]
        assert list(com[1, 1]) == [19.0, 2.0]
        assert list(com[1, 2]) == [11.0, 6.0]

    def test_center_of_mass_dummy_data(self):
        sub = dd.get_distorted_cubic_sublattice()
        sub.find_nearest_neighbors()
        sub.refine_atom_positions_using_center_of_mass(show_progressbar=False)
        positions = [
            [30.0, 30.0],
            [30.0, 50.0],
            [30.0, 70.0],
            [30.0, 90.0],
            [30.0, 110.0],
            [30.0, 130.0],
            [30.0, 150.0],
            [30.0, 170.0],
            [30.0, 190.0],
            [30.0, 210.0],
        ]
        np.testing.assert_almost_equal(
            np.array(sub.atom_positions)[:10, :10], positions, decimal=5
        )


class TestFitAtomPositionsWithGaussianModel:
    def setup_method(self):
        test_data = MakeTestData(100, 100)
        x, y = np.mgrid[10:90:10j, 10:90:10j]
        x, y = x.flatten(), y.flatten()
        sigma, A = 1, 50
        test_data.add_atom_list(x, y, sigma_x=sigma, sigma_y=sigma, amplitude=A)
        self.sublattice = test_data.sublattice
        self.sublattice.find_nearest_neighbors()

    def test_1_atom(self):
        sublattice = self.sublattice
        g_list = afr._fit_atom_positions_with_gaussian_model(
            [sublattice.atom_list[5]], sublattice.image
        )
        assert len(g_list) == 1

    def test_2_atom(self):
        sublattice = self.sublattice
        g_list = afr._fit_atom_positions_with_gaussian_model(
            sublattice.atom_list[5:7], sublattice.image
        )
        assert len(g_list) == 2

    def test_5_atom(self):
        sublattice = self.sublattice
        g_list = afr._fit_atom_positions_with_gaussian_model(
            sublattice.atom_list[5:10], sublattice.image
        )
        assert len(g_list) == 5

    def test_wrong_input_0(self):
        sublattice = self.sublattice
        with pytest.raises(TypeError):
            afr._fit_atom_positions_with_gaussian_model(
                sublattice.atom_list[5], sublattice.image
            )

    def test_wrong_input_1(self):
        sublattice = self.sublattice
        with pytest.raises(TypeError):
            afr._fit_atom_positions_with_gaussian_model(
                [sublattice.atom_list[5:7]], sublattice.image
            )


class TestAtomToGaussianComponent:
    def test_simple(self):
        x, y, sX, sY, r = 7.1, 2.8, 2.1, 3.3, 1.9
        atom_position = Atom_Position(x=x, y=y, sigma_x=sX, sigma_y=sY, rotation=r)
        gaussian = afr._atom_to_gaussian_component(atom_position)
        assert x == gaussian.centre_x.value
        assert y == gaussian.centre_y.value
        assert sX == gaussian.sigma_x.value
        assert sY == gaussian.sigma_y.value
        assert r == gaussian.rotation.value


class TestMakeCircularMask:
    def test_small_radius_1(self):
        imX, imY = 3, 3
        mask = afr._make_circular_mask(1, 1, imX, imY, 1)
        assert mask.size == imX * imY
        assert mask.sum() == 5
        true_index = [[1, 0], [0, 1], [1, 1], [2, 1], [1, 2]]
        false_index = [[0, 0], [0, 2], [2, 0], [2, 2]]
        for index in true_index:
            assert mask[index[0], index[1]]
        for index in false_index:
            assert not mask[index[0], index[1]]

    def test_all_true_mask(self):
        imX, imY = 5, 5
        mask = afr._make_circular_mask(1, 1, imX, imY, 5)
        assert mask.all()
        assert mask.size == imX * imY
        assert mask.sum() == imX * imY

    def test_all_false_mask(self):
        mask = afr._make_circular_mask(10, 10, 5, 5, 3)
        assert not mask.any()


class TestMakeMaskCircleCentre:
    @pytest.mark.parametrize(
        "dtype", ["uint8", "uint16", "uint32", "float16", "float32", "bool"]
    )
    def test_dtypes(self, dtype):
        arr = np.zeros((3, 3), dtype=dtype)
        mask = afr._make_mask_circle_centre(arr, 1)
        np.testing.assert_equal(
            mask,
            np.array([[True, False, True], [False, False, False], [True, False, True]]),
        )

    @pytest.mark.parametrize("shape", [(3, 3), (5, 3), (6, 9)])
    def test_different_arr_shapes(self, shape):
        arr = np.zeros(shape)
        mask = afr._make_mask_circle_centre(arr, 2)
        assert arr.shape == mask.shape

    def test_radius_2(self):
        arr = np.zeros((3, 3))
        mask = afr._make_mask_circle_centre(arr, 2)
        np.testing.assert_array_equal(mask, np.zeros((3, 3), dtype=np.bool))

    def test_wrong_arr_dimensions(self):
        arr = np.zeros((3, 3, 4))
        with pytest.raises(ValueError):
            afr._make_mask_circle_centre(arr, 2)


class TestZeroArrayOutsideCircle:
    @pytest.mark.parametrize(
        "dtype", ["uint8", "uint16", "uint32", "float16", "float32", "bool"]
    )
    def test_dtypes(self, dtype):
        arr = np.ones((3, 3)) * 9
        arr = arr.astype(dtype)
        data = afr.zero_array_outside_circle(arr, 1)
        np.testing.assert_equal(
            data, np.array([[0, 9, 0], [9, 9, 9], [0, 9, 0]], dtype=dtype)
        )

    def test_correct_number_of_ones(self):
        one = np.ones((10, 10))
        assert np.sum(afr.zero_array_outside_circle(one, 3)) == 32

    def test_radius_2(self):
        arr = np.zeros((3, 3))
        mask = afr._make_mask_circle_centre(arr, 2)
        np.testing.assert_equal(mask == 0, True)

    def test_wrong_arr_dimensions(self):
        arr = np.zeros((3, 3, 4))
        with pytest.raises(ValueError):
            afr.zero_array_outside_circle(arr, 2)


class TestFitAtomPositionsGaussian:
    def setup_method(self):
        test_data = MakeTestData(100, 100)
        x, y = np.mgrid[5:95:10j, 5:95:10j]
        x, y = x.flatten(), y.flatten()
        test_data.add_atom_list(x, y)
        sublattice = test_data.sublattice
        sublattice.construct_zone_axes()
        self.sublattice = sublattice
        self.x, self.y = x, y

    def test_one_atoms(self):
        sublattice = self.sublattice
        atom_index = 55
        atom_list = [sublattice.atom_list[atom_index]]
        image_data = sublattice.image
        afr.fit_atom_positions_gaussian(atom_list, image_data)
        assert_allclose(
            sublattice.atom_list[atom_index].pixel_x, self.x[atom_index], rtol=1e-7
        )
        assert_allclose(
            sublattice.atom_list[atom_index].pixel_y, self.y[atom_index], rtol=1e-7
        )

    def test_two_atoms(self):
        sublattice = self.sublattice
        atom_indices = [44, 45]
        atom_list = []
        for index in atom_indices:
            atom_list.append(sublattice.atom_list[index])
        image_data = sublattice.image
        afr.fit_atom_positions_gaussian(atom_list, image_data)
        for atom_index in atom_indices:
            assert_allclose(
                sublattice.atom_list[atom_index].pixel_x, self.x[atom_index], rtol=1e-7
            )
            assert_allclose(
                sublattice.atom_list[atom_index].pixel_y, self.y[atom_index], rtol=1e-7
            )

    def test_four_atoms(self):
        sublattice = self.sublattice
        atom_indices = [35, 36, 45, 46]
        atom_list = []
        for index in atom_indices:
            atom_list.append(sublattice.atom_list[index])
        image_data = sublattice.image
        afr.fit_atom_positions_gaussian(atom_list, image_data)
        for atom_index in atom_indices:
            assert_allclose(
                sublattice.atom_list[atom_index].pixel_x, self.x[atom_index], rtol=1e-7
            )
            assert_allclose(
                sublattice.atom_list[atom_index].pixel_y, self.y[atom_index], rtol=1e-7
            )

    def test_nine_atoms(self):
        sublattice = self.sublattice
        atom_indices = [34, 35, 36, 44, 45, 46, 54, 55, 56]
        atom_list = []
        for index in atom_indices:
            atom_list.append(sublattice.atom_list[index])
        image_data = sublattice.image
        afr.fit_atom_positions_gaussian(atom_list, image_data)
        for atom_index in atom_indices:
            assert_allclose(
                sublattice.atom_list[atom_index].pixel_x, self.x[atom_index], rtol=1e-7
            )
            assert_allclose(
                sublattice.atom_list[atom_index].pixel_y, self.y[atom_index], rtol=1e-7
            )

    def test_wrong_input_none_mask_radius_percent_to_nn(self):
        sublattice = self.sublattice
        with pytest.raises(ValueError):
            afr.fit_atom_positions_gaussian(
                sublattice.atom_list,
                sublattice.image,
                percent_to_nn=None,
                mask_radius=None,
            )


class TestGetAtomPositions:
    def test_find_number_of_columns(self):
        test_data = MakeTestData(50, 50)
        x, y = np.mgrid[5:48:5, 5:48:5]
        x, y = x.flatten(), y.flatten()
        test_data.add_atom_list(x, y)
        peaks = afr.get_atom_positions(test_data.signal, separation=3)
        assert len(peaks) == len(x)

    @pytest.mark.parametrize("separation", [-1000, -1, 0, 0.0, 0.2, 0.9999])
    def test_too_low_separation(self, separation):
        s = dd.get_simple_cubic_signal()
        with pytest.raises(ValueError):
            afr.get_atom_positions(s, separation)

    def test_wrong_dimensions(self):
        s1 = Signal1D(np.random.random(200))
        s3 = Signal2D(np.random.random((4, 200, 200)))
        s4 = Signal2D(np.random.random((3, 4, 200, 200)))
        with pytest.raises(ValueError):
            afr.get_atom_positions(s1, 2)
        with pytest.raises(ValueError):
            afr.get_atom_positions(s3, 2)
        with pytest.raises(ValueError):
            afr.get_atom_positions(s4, 2)


class TestBadFitCondition:
    def setup_method(self):
        t = MakeTestData(40, 40)
        x, y = np.mgrid[5:40:10, 5:40:10]
        x, y = x.flatten(), y.flatten()
        t.add_atom_list(x, y)
        self.sublattice = t.sublattice
        self.sublattice.find_nearest_neighbors()

    def test_initial_position_inside_mask_x(self):
        sublattice = self.sublattice
        atom = [sublattice.atom_list[6]]
        x0 = atom[0].pixel_x
        atom[0].pixel_x += 2
        g = afr._fit_atom_positions_with_gaussian_model(
            atom, sublattice.image, mask_radius=4
        )
        assert_allclose(g[0].centre_x.value, x0, rtol=1e-2)

    def test_initial_position_outside_mask_x(self):
        sublattice = self.sublattice
        atom = [sublattice.atom_list[6]]
        atom[0].pixel_x += 3
        g = afr._fit_atom_positions_with_gaussian_model(
            atom, sublattice.image, mask_radius=2
        )
        assert not g

    def test_initial_position_outside_mask_y(self):
        sublattice = self.sublattice
        atom = [sublattice.atom_list[6]]
        atom[0].pixel_y -= 4
        g = afr._fit_atom_positions_with_gaussian_model(
            atom, sublattice.image, mask_radius=2
        )
        assert not g

    def test_initial_position_outside_mask_xy(self):
        sublattice = self.sublattice
        atom = [sublattice.atom_list[6]]
        atom[0].pixel_y += 3
        atom[0].pixel_x += 3
        g = afr._fit_atom_positions_with_gaussian_model(
            atom, sublattice.image, mask_radius=2
        )
        assert not g

    def test_too_small_percent_to_nn(self):
        sublattice = self.sublattice
        afr._fit_atom_positions_with_gaussian_model(
            [sublattice.atom_list[6]], sublattice.image, percent_to_nn=0.01
        )


class TestFitOutsideImageBounds:
    def test_outside_low_y(self):
        im_x, im_y = 100, 90
        test_data = MakeTestData(im_x, im_y)
        test_data.add_atom(50, -10, amplitude=200, sigma_x=10, sigma_y=10)

        image = test_data.signal.data
        atom_position = Atom_Position(50, 10)
        gaussian_list = afr._fit_atom_positions_with_gaussian_model(
            [atom_position], image, mask_radius=30
        )
        if gaussian_list is not False:
            gaussian = gaussian_list[0]
            assert gaussian.centre_x.value > 0
            assert gaussian.centre_x.value < im_x
            assert gaussian.centre_y.value > 0
            assert gaussian.centre_y.value < im_y

    def test_outside_low_x(self):
        im_x, im_y = 80, 90
        test_data = MakeTestData(im_x, im_y)
        test_data.add_atom(-10, 30, amplitude=200, sigma_x=10, sigma_y=10)

        image = test_data.signal.data
        atom_position = Atom_Position(10, 30)
        gaussian_list = afr._fit_atom_positions_with_gaussian_model(
            [atom_position], image, mask_radius=30
        )
        if gaussian_list is not False:
            gaussian = gaussian_list[0]
            assert gaussian.centre_x.value > 0
            assert gaussian.centre_x.value < im_x
            assert gaussian.centre_y.value > 0
            assert gaussian.centre_y.value < im_y

    def test_outside_high_y(self):
        im_x, im_y = 100, 90
        test_data = MakeTestData(im_x, im_y)
        test_data.add_atom(50, 100, amplitude=200, sigma_x=10, sigma_y=10)

        image = test_data.signal.data
        atom_position = Atom_Position(50, 90)
        gaussian_list = afr._fit_atom_positions_with_gaussian_model(
            [atom_position], image, mask_radius=30
        )
        if gaussian_list is not False:
            gaussian = gaussian_list[0]
            assert gaussian.centre_x.value > 0
            assert gaussian.centre_x.value < im_x
            assert gaussian.centre_y.value > 0
            assert gaussian.centre_y.value < im_y

    def test_outside_high_x(self):
        im_x, im_y = 90, 100
        test_data = MakeTestData(im_x, im_y)
        test_data.add_atom(100, 50, amplitude=200, sigma_x=10, sigma_y=10)

        image = test_data.signal.data
        atom_position = Atom_Position(80, 50)
        gaussian_list = afr._fit_atom_positions_with_gaussian_model(
            [atom_position], image, mask_radius=30
        )
        if gaussian_list is not False:
            gaussian = gaussian_list[0]
            assert gaussian.centre_x.value > 0
            assert gaussian.centre_x.value < im_x
            assert gaussian.centre_y.value > 0
            assert gaussian.centre_y.value < im_y


class TestGetFeatureSeparation:
    def test_simple(self):
        s = dd.get_simple_cubic_signal()
        s_fs = afr.get_feature_separation(s)
        s_fs.plot()

    def test_separation_range(self):
        sr0, sr1 = 10, 15
        s = dd.get_simple_cubic_signal()
        s_fs = afr.get_feature_separation(s, separation_range=(sr0, sr1))
        assert s_fs.axes_manager.navigation_size == (sr1 - sr0)
        assert s_fs.axes_manager.navigation_extent == (sr0, sr1 - 1)

    def test_separation_step(self):
        sr0, sr1, s_step = 10, 16, 2
        s = dd.get_simple_cubic_signal()
        s_fs = afr.get_feature_separation(
            s, separation_range=(sr0, sr1), separation_step=s_step
        )
        assert s_fs.axes_manager.navigation_size == ((sr1 - sr0) / s_step)

    def test_pca_subtract_background_normalize_intensity(self):
        s = dd.get_simple_cubic_signal()
        s_fs = afr.get_feature_separation(
            s, pca=True, subtract_background=True, normalize_intensity=True
        )
        assert hasattr(s_fs, "data")

    def test_dtypes(self):
        dtype_list = [
            "float64",
            "float32",
            "uint64",
            "uint32",
            "uint16",
            "uint8",
            "int64",
            "int32",
            "int16",
            "int8",
        ]
        s = dd.get_simple_cubic_signal()
        s *= 10 ** 9
        for dtype in dtype_list:
            print(dtype)
            s.change_dtype(dtype)
            afr.get_feature_separation(s, separation_range=(10, 15))
        s.change_dtype("float16")
        with pytest.raises(ValueError):
            afr.get_feature_separation(s, separation_range=(10, 15))

    @pytest.mark.parametrize("separation_low", [-1000, -1, 0, 0.0, 0.2, 0.999])
    def test_too_low_separation_low(self, separation_low):
        separation_range = (separation_low, 3)
        s = dd.get_simple_cubic_signal()
        with pytest.raises(ValueError):
            afr.get_feature_separation(s, separation_range)

    @pytest.mark.parametrize("separation_range", [(10, 2), (1000, 2), (2, 1)])
    def test_separation_range_bad(self, separation_range):
        s = dd.get_simple_cubic_signal()
        with pytest.raises(ValueError):
            afr.get_feature_separation(s, separation_range)

    def test_small_input_size_large_separation_range(self):
        # For small images and large separation, no peaks can be returned.
        # This test checks that this doesn't result in an error.
        s = dd.get_simple_cubic_signal().isig[:50.0, :50.0]
        afr.get_feature_separation(s)

    def test_find_no_peaks(self):
        s = dd.get_simple_cubic_signal().isig[:5.0, :5.0]
        with pytest.raises(ValueError):
            afr.get_feature_separation(s)
