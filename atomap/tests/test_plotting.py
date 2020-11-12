import numpy as np
from atomap.atom_position import Atom_Position
from atomap.atom_plane import Atom_Plane
from atomap.atom_lattice import Atom_Lattice
import atomap.plotting as pl


class TestMakeAtomPositionMarkerList:
    def test_make_atom_position_marker_list(self):
        atom_position_list = []
        for i in range(20):
            atom_position = Atom_Position(i, i)
            atom_position_list.append(atom_position)
        marker_list = pl._make_atom_position_marker_list(
            atom_position_list,
            scale=0.2,
            markersize=30,
            color="black",
            add_numbers=True,
        )
        assert len(marker_list) == 40


class TestAtomPlaneMarkerPlotting:
    def setup_method(self):
        atom_lattice = Atom_Lattice()
        atom_plane_list = []
        for i in range(2):
            atom_list = [
                Atom_Position(1, 2),
                Atom_Position(2, 4),
            ]
            zone_vector = (1, 2)
            atom_list[0]._start_atom = [zone_vector]
            atom_list[1]._end_atom = [zone_vector]
            atom_plane = Atom_Plane(atom_list, (1, 2), atom_lattice)
            atom_plane_list.append(atom_plane)
        self.atom_plane_list = atom_plane_list

    def test_single_atom_plane_marker(self):
        atom_plane = self.atom_plane_list[0]
        marker_list = pl._make_single_atom_plane_marker_list(atom_plane)
        assert len(marker_list) == 1

    def test_make_atom_planes_marker_list_no_number(self):
        atom_plane_list = self.atom_plane_list
        marker_list = pl._make_atom_planes_marker_list(
            atom_plane_list, add_numbers=False
        )
        assert len(marker_list) == 2

    def test_make_atom_planes_marker_list(self):
        atom_plane_list = self.atom_plane_list
        marker_list = pl._make_atom_planes_marker_list(
            atom_plane_list, add_numbers=True
        )
        assert len(marker_list) == 4


class TestPlotVectorField:
    def test_simple(self):
        x_list, y_list = np.arange(100), np.arange(100, 200)
        xr_list, yr_list = np.arange(200, 300), np.arange(300, 400)
        pl.plot_vector_field(x_list, y_list, xr_list, yr_list)

    def test_save_argument(self):
        x_list, y_list = np.arange(100), np.arange(100, 200)
        xr_list, yr_list = np.arange(200, 300), np.arange(300, 400)
        pl.plot_vector_field(x_list, y_list, xr_list, yr_list, save=True)


class TestMakeFigureScatterPointOnImage:
    def test_simple(self):
        image = np.random.random((100, 100))
        x, y = np.arange(50), np.arange(50)
        z = np.random.randint(1, 9, size=50)
        fig = pl._make_figure_scatter_point_on_image(image, x, y, z)
        fig.show()

    def test_cmap(self):
        image = np.random.random((100, 100))
        x, y = np.arange(50), np.arange(50)
        z = np.random.randint(1, 9, size=50)
        pl._make_figure_scatter_point_on_image(image, x, y, z, cmap="inferno")

    def test_vmin_vmax(self):
        image = np.random.random((100, 100))
        x, y = np.arange(50), np.arange(50)
        z = np.random.randint(1, 9, size=50)
        pl._make_figure_scatter_point_on_image(image, x, y, z, vmin=-10, vmax=-2)
