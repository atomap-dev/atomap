from pytest import approx, mark
import numpy as np
from numpy import pi
import math
from atomap.atom_position import Atom_Position
import atomap.testing_tools as tt
import atomap.dummy_data as dd


class TestCreateAtomPositionObject:
    def test_position(self):
        atom_x, atom_y = 10, 20
        atom_position = Atom_Position(atom_x, atom_y)
        assert atom_position.pixel_x == 10
        assert atom_position.pixel_y == 20

    def test_sigma(self):
        sigma_x, sigma_y = 2, 3
        atom_position = Atom_Position(1, 2, sigma_x=sigma_x, sigma_y=sigma_y)
        assert atom_position.sigma_x == sigma_x
        assert atom_position.sigma_y == sigma_y

        sigma_x, sigma_y = -5, -6
        atom_position = Atom_Position(1, 2, sigma_x=sigma_x, sigma_y=sigma_y)
        assert atom_position.sigma_x == abs(sigma_x)
        assert atom_position.sigma_y == abs(sigma_y)

    def test_amplitude(self):
        amplitude = 30
        atom_position = Atom_Position(1, 2, amplitude=amplitude)
        assert atom_position.amplitude_gaussian == amplitude

    def test_rotation(self):
        rotation0 = 0.0
        atom_position = Atom_Position(1, 2, rotation=rotation0)
        assert atom_position.rotation == rotation0

        rotation1 = math.pi / 2
        atom_position = Atom_Position(1, 2, rotation=rotation1)
        assert atom_position.rotation == rotation1

        rotation2 = math.pi
        atom_position = Atom_Position(1, 2, rotation=rotation2)
        assert atom_position.rotation == 0

        rotation3 = math.pi * 3 / 2
        atom_position = Atom_Position(1, 2, rotation=rotation3)
        assert atom_position.rotation == math.pi / 2

        rotation4 = math.pi * 2
        atom_position = Atom_Position(1, 2, rotation=rotation4)
        assert atom_position.rotation == 0


class TestAtomPositionObjectTools:
    def setup_method(self):
        self.atom_position = Atom_Position(1, 1)

    def test_get_atom_angle(self):
        atom_position0 = Atom_Position(1, 2)
        atom_position1 = Atom_Position(3, 1)
        atom_position2 = Atom_Position(1, 0)
        atom_position3 = Atom_Position(5, 1)
        atom_position4 = Atom_Position(2, 2)

        angle90 = self.atom_position.get_angle_between_atoms(
            atom_position0, atom_position1
        )
        angle180 = self.atom_position.get_angle_between_atoms(
            atom_position0, atom_position2
        )
        angle0 = self.atom_position.get_angle_between_atoms(
            atom_position1, atom_position3
        )
        angle45 = self.atom_position.get_angle_between_atoms(
            atom_position1, atom_position4
        )

        assert approx(angle90) == pi / 2
        assert approx(angle180) == pi
        assert approx(angle0) == 0
        assert approx(angle45) == pi / 4

    def test_as_gaussian(self):
        x, y, sx, sy, A, r = 10.0, 5.0, 2.0, 3.5, 9.9, 1.5
        atom_position = Atom_Position(
            x=x, y=y, sigma_x=sx, sigma_y=sy, amplitude=A, rotation=r
        )
        g = atom_position.as_gaussian()
        assert g.centre_x.value == x
        assert g.centre_y.value == y
        assert g.sigma_x.value == sx
        assert g.sigma_y.value == sy
        assert g.A.value == A
        assert g.rotation.value == r


class TestGetCenterPositionCom:
    def test_mask_radius(self):
        atom_position0 = Atom_Position(15, 20)
        atom_position1 = Atom_Position(15, 20)
        image = np.random.randint(1000, size=(30, 30))
        x0, y0 = atom_position0._get_center_position_com(image, mask_radius=5)
        x1, y1 = atom_position1._get_center_position_com(image, mask_radius=10)
        assert x0 != x1
        assert y0 != y1


class TestAtomPositionRefine:
    def test_center_of_mass_mask_radius(self):
        x, y, sx, sy = 15, 20, 2, 2
        test_data = tt.MakeTestData(50, 50)
        test_data.add_atom(x, y, sx, sy)
        sublattice = test_data.sublattice
        atom = sublattice.atom_list[0]
        image_data = test_data.signal.data
        atom.refine_position_using_center_of_mass(image_data, mask_radius=5)
        assert atom.pixel_x == approx(x)
        assert atom.pixel_y == approx(y)

    def test_2d_gaussian_mask_radius(self):
        x, y, sx, sy = 15, 20, 2, 2
        test_data = tt.MakeTestData(50, 50)
        test_data.add_atom(x, y, sx, sy)
        sublattice = test_data.sublattice
        atom = sublattice.atom_list[0]
        image_data = test_data.signal.data
        atom.refine_position_using_2d_gaussian(image_data, mask_radius=10)
        assert atom.pixel_x == approx(x)
        assert atom.pixel_y == approx(y)
        assert atom.sigma_x == approx(sx, rel=1e-4)
        assert atom.sigma_y == approx(sy, rel=1e-4)


class TestAtomPositionRefinePositionFlag:
    def test_one_atom_center_of_mass(self):
        test_data = tt.MakeTestData(20, 20)
        x, y = 10, 15
        test_data.add_atom(x, y)
        sublattice = test_data.sublattice
        atom = sublattice.atom_list[0]
        atom.pixel_x += 1
        atom.refine_position = False
        sublattice.refine_atom_positions_using_center_of_mass(mask_radius=5)
        assert atom.pixel_x == x + 1
        assert atom.pixel_y == y
        atom.refine_position = True
        sublattice.refine_atom_positions_using_center_of_mass(mask_radius=5)
        assert approx(atom.pixel_x, abs=0.001) == x
        assert approx(atom.pixel_y, abs=0.001) == y

    def test_one_atom_2d_gaussian_refine(self):
        test_data = tt.MakeTestData(20, 20)
        x, y = 10, 15
        test_data.add_atom(x, y)
        sublattice = test_data.sublattice
        atom = sublattice.atom_list[0]
        atom.pixel_x += 1
        atom.refine_position = False
        sublattice.refine_atom_positions_using_2d_gaussian(mask_radius=5)
        assert atom.pixel_x == x + 1
        assert atom.pixel_y == y
        atom.refine_position = True
        sublattice.refine_atom_positions_using_2d_gaussian(mask_radius=5)
        assert approx(atom.pixel_x) == x
        assert approx(atom.pixel_y) == y

    def test_many_atoms(self):
        sublattice = dd.get_simple_cubic_sublattice(image_noise=True)
        atom = sublattice.atom_list[0]
        atom.refine_position = False
        x_pos_orig = np.array(sublattice.x_position)
        y_pos_orig = np.array(sublattice.y_position)
        sublattice.find_nearest_neighbors()
        sublattice.refine_atom_positions_using_2d_gaussian()
        assert atom.pixel_x == x_pos_orig[0]
        assert atom.pixel_y == x_pos_orig[1]
        assert not (x_pos_orig[1:] == sublattice.x_position[1:]).any()
        assert not (y_pos_orig[1:] == sublattice.y_position[1:]).any()
        sublattice.refine_atom_positions_using_2d_gaussian()
        assert atom.pixel_x == x_pos_orig[0]
        assert atom.pixel_y == x_pos_orig[1]
        assert not (x_pos_orig[1:] == sublattice.x_position[1:]).any()
        assert not (y_pos_orig[1:] == sublattice.y_position[1:]).any()


class TestAtomPositionGetAtomSlice:
    @mark.parametrize("quantile", [1, 2, 3, 4])
    def test_sigma_quantile(self, quantile):
        x, y, sx, sy, sigma_quantile = 25, 30, 2, 4, quantile
        atom_position = Atom_Position(x=x, y=y, sigma_x=sx, sigma_y=sy)
        slice_y, slice_x = atom_position._get_atom_slice(
            100, 100, sigma_quantile=sigma_quantile
        )
        smax = max(sx, sy)
        assert slice_x.start == x - smax * sigma_quantile
        assert slice_x.stop == x + smax * sigma_quantile
        assert slice_y.start == y - smax * sigma_quantile
        assert slice_y.stop == y + smax * sigma_quantile

    def test_image_border_x_0(self):
        x, y, sigma, sigma_quantile = 5, 50, 3, 3
        atom_position = Atom_Position(x, y, sigma_x=sigma, sigma_y=sigma)
        slice_y, slice_x = atom_position._get_atom_slice(
            100, 100, sigma_quantile=sigma_quantile
        )
        assert slice_x.start == 0
        assert slice_x.stop == x + sigma * sigma_quantile
        assert slice_y.start == y - sigma * sigma_quantile
        assert slice_y.stop == y + sigma * sigma_quantile

    def test_image_border_y_0(self):
        x, y, sigma, sigma_quantile = 50, 0, 3, 3
        atom_position = Atom_Position(x, y, sigma_x=sigma, sigma_y=sigma)
        slice_y, slice_x = atom_position._get_atom_slice(
            100, 100, sigma_quantile=sigma_quantile
        )
        assert slice_x.start == x - sigma * sigma_quantile
        assert slice_x.stop == x + sigma * sigma_quantile
        assert slice_y.start == 0
        assert slice_y.stop == y + sigma * sigma_quantile

    @mark.parametrize("im_x", [96, 97, 98])
    def test_image_border_x_max(self, im_x):
        x, y, sigma, sigma_quantile = 95, 50, 3, 3
        atom_position = Atom_Position(x, y, sigma_x=sigma, sigma_y=sigma)
        slice_y, slice_x = atom_position._get_atom_slice(
            im_x, 100, sigma_quantile=sigma_quantile
        )
        assert slice_x.start == x - sigma * sigma_quantile
        assert slice_x.stop == im_x
        assert slice_y.start == y - sigma * sigma_quantile
        assert slice_y.stop == y + sigma * sigma_quantile

    @mark.parametrize("im_y", [96, 97, 98])
    def test_image_border_y_max(self, im_y):
        x, y, sigma, sigma_quantile = 50, 95, 3, 3
        atom_position = Atom_Position(x, y, sigma_x=sigma, sigma_y=sigma)
        slice_y, slice_x = atom_position._get_atom_slice(
            100, im_y, sigma_quantile=sigma_quantile
        )
        assert slice_x.start == x - sigma * sigma_quantile
        assert slice_x.stop == x + sigma * sigma_quantile
        assert slice_y.start == y - sigma * sigma_quantile
        assert slice_y.stop == im_y


class TestCalculateAtomPositionIntensity:
    def test_calculate_max_intensity(self):
        test_data = tt.MakeTestData(110, 110)
        x, y = np.mgrid[5:105:5j, 5:105:5j]
        x, y = x.flatten(), y.flatten()
        A = [1] * len(x)
        test_data.add_atom_list(x=x, y=y, amplitude=A)
        sublattice = test_data.sublattice
        sublattice.find_nearest_neighbors()
        sublattice.image /= sublattice.image.max()

        max_intensities = []
        for atom in sublattice.atom_list:
            max_intensities.append(atom.calculate_max_intensity(sublattice.image))

        assert approx(max_intensities) == A

    def test_calculate_min_intensity(self):
        test_data = tt.MakeTestData(110, 110)
        x, y = np.mgrid[5:105:5j, 5:105:5j]
        x, y = x.flatten(), y.flatten()
        A = [1] * len(x)
        A_min = [0] * len(x)
        test_data.add_atom_list(x=x, y=y, amplitude=A)
        sublattice = test_data.sublattice
        sublattice.find_nearest_neighbors()
        sublattice.image /= sublattice.image.max()

        min_intensities = []
        for atom in sublattice.atom_list:
            min_intensities.append(atom.calculate_min_intensity(sublattice.image))

        assert approx(min_intensities) == A_min


class TestGetImageSliceAroundAtom:
    def test_simple(self):
        image_data = np.random.random((50, 50))
        x, y, distance = 10, 15, 4
        atom = Atom_Position(x, y)
        distance = 4
        image_slice, x0, y0 = atom._get_image_slice_around_atom(image_data, distance)
        assert image_slice.shape == (9, 9)
        assert x0 == (x - distance)
        assert y0 == (y - distance)

    def test_close_to_edge_x0(self):
        image_data = np.random.random((50, 50))
        x, y, distance = 1, 15, 4
        atom = Atom_Position(x, y)
        image_slice, x0, y0 = atom._get_image_slice_around_atom(image_data, distance)
        assert image_slice.shape == (9, 6)
        assert x0 == 0
        assert y0 == (y - distance)

    def test_close_to_edge_y0(self):
        image_data = np.random.random((50, 50))
        x, y, distance = 20, 1, 5
        atom = Atom_Position(x, y)
        image_slice, x0, y0 = atom._get_image_slice_around_atom(image_data, distance)
        assert image_slice.shape == (7, 11)
        assert x0 == (x - distance)
        assert y0 == 0

    def test_close_to_edge_x1(self):
        image_data = np.random.random((50, 50))
        x, y, distance = 48, 15, 4
        atom = Atom_Position(x, y)
        image_slice, x0, y0 = atom._get_image_slice_around_atom(image_data, distance)
        assert image_slice.shape == (9, 6)
        assert x0 == (x - distance)
        assert y0 == (y - distance)

    def test_close_to_edge_y1(self):
        image_data = np.random.random((50, 50))
        x, y, distance = 20, 49, 4
        atom = Atom_Position(x, y)
        image_slice, x0, y0 = atom._get_image_slice_around_atom(image_data, distance)
        assert image_slice.shape == (5, 9)
        assert x0 == (x - distance)
        assert y0 == (y - distance)

    def test_close_to_edge_corner(self):
        image_data = np.random.random((50, 60))
        x, y, distance = 0, 0, 9
        atom = Atom_Position(x, y)
        image_slice, x0, y0 = atom._get_image_slice_around_atom(image_data, distance)
        assert image_slice.shape == (10, 10)
        assert x0 == 0
        assert y0 == 0

    def test_value(self):
        image_data = np.zeros((100, 100))
        x, y, distance, value = 20, 49, 4, 10
        image_data[y, x] = value
        atom = Atom_Position(x, y)
        image_slice, x0, y0 = atom._get_image_slice_around_atom(image_data, distance)
        assert image_slice[distance, distance] == value


class TestEstimateLocalScanningDistortion:
    def test_flat_image(self):
        image_data = np.ones((100, 100)) * 10
        atom = Atom_Position(20, 10)
        distortion_x, distortion_y = atom.estimate_local_scanning_distortion(
            image_data, radius=6, edge_skip=2
        )
        assert approx(distortion_x, 0)
        assert approx(distortion_y, 0)

    def test_values(self):
        image_data = np.zeros((100, 100))
        image_data[77, 20] = 10
        image_data[76, 21] = 10
        image_data[75, 20] = 10
        image_data[74, 21] = 10
        image_data[73, 20] = 10
        atom = Atom_Position(20.5, 75)
        distortion_x, distortion_y = atom.estimate_local_scanning_distortion(
            image_data, radius=4, edge_skip=2
        )
        assert approx(distortion_y, np.std((20, 21, 20, 21, 20)))

    def test_elliptical_atom(self):
        image = np.zeros((100, 100))
        atom = Atom_Position(17, 75)
        image[77, 15] = 10
        image[76, 16] = 10
        image[75, 17] = 10
        image[74, 18] = 10
        image[73, 19] = 10
        scanning_distortions = atom.estimate_local_scanning_distortion(
            image, radius=4, edge_skip=2
        )
        assert approx(scanning_distortions) == (0.0, 0.0)

    def test_radius(self):
        image = np.zeros((100, 100))
        image[79, 17] = 100
        image[77, 15] = 10
        image[76, 16] = 10
        image[75, 17] = 10
        image[74, 18] = 10
        image[73, 19] = 10
        atom = Atom_Position(17, 75)
        scanning_distortions = atom.estimate_local_scanning_distortion(
            image, radius=2, edge_skip=0
        )
        assert approx(scanning_distortions) == (0.0, 0.0)
        scanning_distortions = atom.estimate_local_scanning_distortion(
            image, radius=3, edge_skip=0
        )
        assert approx(scanning_distortions) == (0.0, 0.0)
        scanning_distortions = atom.estimate_local_scanning_distortion(
            image, radius=4, edge_skip=0
        )
        assert approx(scanning_distortions) != (0.0, 0.0)

    def test_edge_skip_y(self):
        image = np.zeros((100, 100))
        image[79, 17] = 100
        image[77, 15] = 10
        image[76, 16] = 10
        image[75, 17] = 10
        image[74, 18] = 10
        image[73, 19] = 10
        atom = Atom_Position(17, 75)
        scanning_distortions = atom.estimate_local_scanning_distortion(
            image, radius=4, edge_skip=0
        )
        assert approx(scanning_distortions[0]) != 0.0
        assert approx(scanning_distortions[1]) != 0.0
        scanning_distortions = atom.estimate_local_scanning_distortion(
            image, radius=4, edge_skip=1
        )
        assert approx(scanning_distortions[0]) != 0.0
        assert approx(scanning_distortions[1]) == 0.0

    def test_edge_skip_x(self):
        image = np.zeros((100, 100))
        image[75, 13] = 100
        image[77, 15] = 10
        image[76, 16] = 10
        image[75, 17] = 10
        image[74, 18] = 10
        image[73, 19] = 10
        atom = Atom_Position(17, 75)
        scanning_distortions = atom.estimate_local_scanning_distortion(
            image, radius=4, edge_skip=0
        )
        assert approx(scanning_distortions[0]) != 0.0
        assert approx(scanning_distortions[1]) != 0.0
        scanning_distortions = atom.estimate_local_scanning_distortion(
            image, radius=4, edge_skip=1
        )
        assert approx(scanning_distortions[0]) == 0.0
        assert approx(scanning_distortions[1]) != 0.0


class TestAtomPositionElementInfo:
    def test_simple(self):
        atom = Atom_Position(20, 10)
        atom.set_element_info("C", [0.0, 0.5])
        assert atom.element_info[0.0] == "C"

        atom.set_element_info(["C", "O"], [0.0, 0.5])
        assert atom.element_info[0.0] == "C"
        assert atom.element_info[0.5] == "O"
