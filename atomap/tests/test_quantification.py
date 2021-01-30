import numpy as np
import math
import atomap.quantification as quant
from atomap.example_data import get_detector_image_signal
from atomap.dummy_data import get_simple_cubic_signal
import atomap.atom_finding_refining as atom_finding
from atomap.sublattice import Sublattice
import atomap.testing_tools as tt


class TestDetectorNormalisation:
    def test_centered_distance_matrix(self):
        s = quant.centered_distance_matrix((32, 32), np.zeros((64, 64)))
        assert s[32, 32] == 1
        assert s[63, 63] == np.sqrt((63 - 31) ** 2 + (63 - 32) ** 2)

    def test_detector_threshold(self):
        det_image = get_detector_image_signal()
        threshold_image = quant._detector_threshold(det_image.data)
        assert not (np.sum(threshold_image) == 0)
        assert det_image.data.shape == threshold_image.shape

    def test_radial_profile(self):
        det_image = get_detector_image_signal()
        profile = quant._radial_profile(det_image.data, (256, 256))
        assert len(np.shape(profile)) == 1
        assert np.shape(profile)[0] == math.ceil(math.sqrt(2) * 256)

    def test_detector_normalisation(self):
        det_image = get_detector_image_signal()
        img = get_simple_cubic_signal(image_noise=True)
        img = (img) * 300000 + 4000
        image_normalised = quant.detector_normalisation(img, det_image, 60)
        assert image_normalised.data.max() < 1
        assert image_normalised.data.shape == img.data.shape

    def test_func(self):
        result = quant._func(4, 2, 0.5, 5)
        assert result == 6

    def test_find_flux_limits_running(self):
        flux1 = quant.centered_distance_matrix((63, 63), np.zeros((128, 128)))
        (profiler, flux_profile) = quant.find_flux_limits(100 - flux1, 25)
        assert len(flux_profile) == math.ceil((64 ** 2 + 64 ** 2) ** 0.5)


class TestStatisticalQuant:
    def setup_method(self):
        self.tdata = tt.MakeTestData(200, 200)

        for i in range(4):
            x, y = np.mgrid[60 * i : (i + 1) * 60 : 15, 10:200:15]
            x, y = x.flatten(), y.flatten()
            self.tdata.add_atom_list(
                x, y, sigma_x=2, sigma_y=2, amplitude=(i + 1) * 20, rotation=0.4
            )
        self.tdata.add_image_noise(sigma=0.02)

        atom_positions = atom_finding.get_atom_positions(
            self.tdata.signal, 8, threshold_rel=0.1
        )

        self.sublattice = Sublattice(atom_positions, self.tdata.signal.data)
        self.sublattice.construct_zone_axes()
        self.sublattice.refine_atom_positions_using_2d_gaussian(self.sublattice.image)

    def test_quant_criteria(self):
        quant.get_statistical_quant_criteria([self.sublattice], 10)

    def test_plot_fitted_hist(self):
        models = quant.get_statistical_quant_criteria([self.sublattice], 10)
        model = models[3]

        intensities = [
            2 * np.pi * atom.amplitude_gaussian * atom.sigma_x * atom.sigma_y
            for atom in self.sublattice.atom_list
        ]
        int_array = np.asarray(intensities)
        int_array = int_array.reshape(-1, 1)

        sort_indices = model.means_.argsort(axis=0)

        labels = model.predict(int_array)

        dic = {}
        for i in range(4):
            dic[int(sort_indices[i])] = i

        sorted_labels = np.copy(labels)
        for k, v in dic.items():
            sorted_labels[labels == k] = v

        from matplotlib import cm

        x = np.linspace(0.0, 1.0, 4)
        rgb = cm.get_cmap("viridis")(x)[np.newaxis, :, :3].tolist()

        quant._plot_fitted_hist(int_array, model, rgb, sort_indices)

    def test_statistical_method(self):
        models = quant.get_statistical_quant_criteria([self.sublattice], 10)
        atom_lattice = quant.statistical_quant(
            self.sublattice, models[3], 4, "C", 3.5, plot=False
        )

        assert len(atom_lattice.sublattice_list[0].atom_list) == 39
        assert len(atom_lattice.sublattice_list[1].atom_list) == 52
        assert len(atom_lattice.sublattice_list[2].atom_list) == 52
        assert len(atom_lattice.sublattice_list[3].atom_list) == 13

        # confirm sublattice.element_info has been assigned correctly
        assert self.sublattice.atom_list[0].element_info == {
            1.75: "C",
            5.25: "C",
            8.75: "C",
            12.25: "C",
        }
        assert len(self.sublattice.atom_list[0].element_info) == 4
        assert self.sublattice.atom_list[0].element_info[5.25] == "C"

    def test_z_ordering(self):
        sublattice = self.sublattice
        models = quant.get_statistical_quant_criteria([sublattice], 10)

        kwargs = {
            "sublattice": sublattice,
            "model": models[3],
            "max_atom_nums": 44,
            "element": "C",
            "z_spacing": 3.5,
            "plot": False,
        }
        quant.statistical_quant(**kwargs, z_ordering="top")
        pos_top = list(sublattice.atom_list[-1].element_info)[0]

        quant.statistical_quant(**kwargs, z_ordering="center")
        pos_center = list(sublattice.atom_list[-1].element_info)[0]

        quant.statistical_quant(**kwargs, z_ordering="bottom")
        pos_bottom = list(sublattice.atom_list[-1].element_info)[0]

        assert pos_top > pos_center
        assert pos_center > pos_bottom
