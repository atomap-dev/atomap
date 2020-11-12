import numpy as np
from atomap.sublattice import Sublattice
import atomap.stats as st


class TestStats:
    def setup_method(self):
        self.atoms_N = 10
        image_data = np.arange(10000).reshape(100, 100)
        peaks = np.arange(20).reshape(self.atoms_N, 2)
        sublattice = Sublattice(peaks, image_data)
        sublattice.original_image = image_data
        for atom in sublattice.atom_list:
            atom.sigma_x = 2.0
            atom.sigma_y = 2.0
            atom.amplitude_gaussian = 10.0
            atom.amplitude_max_intensity = 10.0
        self.sublattice = sublattice

    def test_plot_amplitude_sigma_hist2d(self):
        st.plot_amplitude_sigma_hist2d(self.sublattice)

    def test_plot_atom_column_hist_amplitude_gauss2d_maps(self):
        st.plot_atom_column_hist_amplitude_gauss2d_maps(self.sublattice)

    def test_plot_atom_column_histogram_sigma(self):
        st.plot_atom_column_histogram_sigma(self.sublattice)

    def test_plot_atom_column_histogram_amplitude_gauss2d(self):
        st.plot_atom_column_histogram_amplitude_gauss2d(self.sublattice)

    def test_plot_atom_column_histogram_max_intensity(self):
        st.plot_atom_column_histogram_max_intensity(self.sublattice)

    def test_plot_amplitude_sigma_scatter(self):
        st.plot_amplitude_sigma_scatter(self.sublattice)

    def test_get_atom_list_atom_sigma_range(self):
        atom_list = st.get_atom_list_atom_sigma_range(self.sublattice, (1.0, 3.0))
        assert len(atom_list) == self.atoms_N
