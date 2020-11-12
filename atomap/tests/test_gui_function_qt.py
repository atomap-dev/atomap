from pytest import approx
import numpy as np
import matplotlib.pyplot as plt
from hyperspy.signals import Signal2D
import atomap.initial_position_finding as ipf
from atomap.sublattice import Sublattice
import atomap.testing_tools as tt
import atomap.gui_classes as agc


class TestAddAtomAdderRemoving:
    def test_no_atoms_input(self):
        data = np.random.random((200, 200))
        peaks = ipf.add_atoms_with_gui(data)
        fig = plt.figure(1)
        x, y = fig.axes[0].transData.transform((100, 100))
        assert len(peaks) == 0
        fig.canvas.button_press_event(x, y, 1)
        assert len(peaks) == 1
        fig.canvas.button_press_event(x, y, 1)
        assert len(peaks) == 0

    def test_atoms_input(self):
        data = np.random.random((50, 200))
        peaks = ipf.add_atoms_with_gui(
            data,
            [
                [120, 20],
                [120, 30],
            ],
        )
        fig = plt.figure(1)
        x0, y0 = fig.axes[0].transData.transform((100, 10))
        assert len(peaks) == 2
        fig.canvas.button_press_event(x0, y0, 1)
        assert len(peaks) == 3
        x1, y1 = fig.axes[0].transData.transform((120, 30))
        fig.canvas.button_press_event(x1, y1, 1)
        assert len(peaks) == 2
        assert peaks[0] == approx([120, 20])
        assert peaks[1] == approx([100, 10])

    def test_many_atoms_input(self):
        data = np.random.random((500, 1000))
        x, y = np.mgrid[10:990:100j, 10:490:100j]
        x, y = x.flatten(), y.flatten()
        positions = np.stack((x, y)).T
        peaks = ipf.add_atoms_with_gui(data, positions)
        assert len(peaks) == len(positions)
        fig = plt.figure(1)
        x0, y0 = fig.axes[0].transData.transform((peaks[0][0], peaks[0][1]))
        fig.canvas.button_press_event(x0, y0, 1)
        assert len(peaks) == len(positions) - 1

    def test_distance_threshold(self):
        data = np.random.random((50, 200))
        peaks0 = ipf.add_atoms_with_gui(data, distance_threshold=10)
        fig0 = plt.figure(1)
        x00, y00 = fig0.axes[0].transData.transform((100, 10))
        fig0.canvas.button_press_event(x00, y00, 1)
        assert len(peaks0) == 1
        x01, y01 = fig0.axes[0].transData.transform((109, 19))
        fig0.canvas.button_press_event(x01, y01, 1)
        assert len(peaks0) == 0
        plt.close(fig0)

        peaks1 = ipf.add_atoms_with_gui(data, distance_threshold=4)
        fig1 = plt.figure(1)
        x10, y10 = fig1.axes[0].transData.transform((100, 10))
        fig1.canvas.button_press_event(x10, y10, 1)
        assert len(peaks1) == 1
        x11, y11 = fig1.axes[0].transData.transform((109, 19))
        fig1.canvas.button_press_event(x11, y11, 1)
        assert len(peaks1) == 2
        plt.close(fig1)

    def test_linear_norm(self):
        data = np.random.random((200, 200))
        peaks = ipf.add_atoms_with_gui(data, norm="linear")
        fig = plt.figure(1)
        x, y = fig.axes[0].transData.transform((100, 100))
        assert len(peaks) == 0
        fig.canvas.button_press_event(x, y, 1)
        assert len(peaks) == 1
        fig.canvas.button_press_event(x, y, 1)
        assert len(peaks) == 0

    def test_log_norm(self):
        data = np.random.random((200, 200))
        peaks = ipf.add_atoms_with_gui(data, norm="log")
        fig = plt.figure(1)
        x, y = fig.axes[0].transData.transform((100, 100))
        assert len(peaks) == 0
        fig.canvas.button_press_event(x, y, 1)
        assert len(peaks) == 1
        fig.canvas.button_press_event(x, y, 1)
        assert len(peaks) == 0

    def test_log_norm_negative_image_values(self):
        data = np.random.random((200, 200)) - 10
        ipf.add_atoms_with_gui(data, norm="log")

    def test_signal_input(self):
        s = Signal2D(np.random.random((200, 200)))
        ipf.add_atoms_with_gui(s, norm="log")
        ipf.add_atoms_with_gui(s, norm="linear")


class TestToggleAtomRefinePosition:
    def test_toggle_one(self):
        atom_position_list = [[10, 10], [10, 20]]
        sublattice = Sublattice(atom_position_list, np.zeros((30, 30)))
        sublattice.toggle_atom_refine_position_with_gui()
        fig = plt.figure(1)
        x, y = fig.axes[0].transData.transform((10, 20))
        assert sublattice.atom_list[1].refine_position
        fig.canvas.button_press_event(x, y, 1)
        assert not sublattice.atom_list[1].refine_position
        fig.canvas.button_press_event(x, y, 1)
        assert sublattice.atom_list[1].refine_position

    def test_toggle_all(self):
        atom_position_list = [[10, 10], [10, 20]]
        sublattice = Sublattice(atom_position_list, np.zeros((30, 30)))
        sublattice.toggle_atom_refine_position_with_gui()
        fig = plt.figure(1)
        x0, y0 = fig.axes[0].transData.transform((10, 10))
        x1, y1 = fig.axes[0].transData.transform((10, 20))
        fig.canvas.button_press_event(x0, y0, 1)
        assert not sublattice.atom_list[0].refine_position
        fig.canvas.button_press_event(x1, y1, 1)
        assert not sublattice.atom_list[1].refine_position
        fig.canvas.button_press_event(x0, y0, 1)
        assert sublattice.atom_list[0].refine_position
        fig.canvas.button_press_event(x1, y1, 1)
        assert sublattice.atom_list[1].refine_position

    def test_with_fitting(self):
        x_pos, y_pos = [[10, 10], [10, 20]]
        delta_pos = 1
        test_data = tt.MakeTestData(30, 30)
        test_data.add_atom_list(x_pos, y_pos)
        sublattice = test_data.sublattice
        atom0, atom1 = sublattice.atom_list
        atom0.pixel_x += delta_pos
        atom0.pixel_y += delta_pos
        atom1.pixel_x += delta_pos
        atom1.pixel_y += delta_pos
        sublattice.toggle_atom_refine_position_with_gui()
        fig = plt.figure(1)
        x0, y0 = fig.axes[0].transData.transform(
            (x_pos[0] + delta_pos, y_pos[0] + delta_pos)
        )
        fig.canvas.button_press_event(x0, y0, 1)
        print(atom0.refine_position)
        print(atom1.refine_position)
        sublattice.refine_atom_positions_using_center_of_mass(mask_radius=4)
        assert atom0.pixel_x == (x_pos[0] + delta_pos)
        assert atom0.pixel_y == (y_pos[0] + delta_pos)
        assert atom1.pixel_x != (x_pos[1] + delta_pos)
        assert atom1.pixel_y != (y_pos[1] + delta_pos)


class TestSelectAtomsWithGui:
    def test_select_one_atom(self):
        image = np.random.random((200, 200))
        atom_positions = [[10, 20], [50, 50]]
        atom_selector = agc.GetAtomSelection(image, atom_positions)
        fig = atom_selector.fig
        poly = atom_selector.poly
        position_list = [[5, 5], [5, 25], [25, 25], [25, 5], [5, 5]]
        tt._do_several_move_press_release_event(fig, poly, position_list)
        atom_positions_selected = atom_selector.atom_positions_selected
        assert len(atom_positions_selected) == 1
        assert atom_positions_selected[0] == [10, 20]

    def test_select_one_atom_invert_selection(self):
        image = np.random.random((200, 200))
        atom_positions = [[10, 20], [50, 50]]
        atom_selector = agc.GetAtomSelection(
            image, atom_positions, invert_selection=True
        )
        fig = atom_selector.fig
        poly = atom_selector.poly
        position_list = [[5, 5], [5, 25], [25, 25], [25, 5], [5, 5]]
        tt._do_several_move_press_release_event(fig, poly, position_list)
        atom_positions_selected = atom_selector.atom_positions_selected
        assert len(atom_positions_selected) == 1
        assert atom_positions_selected[0] == [50, 50]

    def test_select_no_atom(self):
        image = np.random.random((200, 200))
        atom_positions = [[10, 20], [50, 50]]
        atom_selector = agc.GetAtomSelection(image, atom_positions)
        fig = atom_selector.fig
        poly = atom_selector.poly
        position_list = [[5, 5], [5, 8], [8, 8], [8, 5], [5, 5]]
        tt._do_several_move_press_release_event(fig, poly, position_list)
        atom_positions_selected = atom_selector.atom_positions_selected
        assert len(atom_positions_selected) == 0

    def test_non_interactive(self):
        image = np.random.random((200, 200))
        atom_positions = [[10, 20], [50, 50]]
        verts = [(5, 5), (5, 30), (20, 30), (20, 5)]
        atom_positions_selected = ipf.select_atoms_with_gui(
            image, atom_positions, verts=verts
        )
        assert len(atom_positions_selected) == 1
        assert (atom_positions_selected[0] == [10, 20]).all()

        verts = [(5, 5), (5, 60), (60, 60), (60, 5)]
        atom_positions_selected = ipf.select_atoms_with_gui(
            image, atom_positions, verts=verts
        )
        assert len(atom_positions_selected) == 2
        assert (atom_positions_selected[0] == [10, 20]).all()
        assert (atom_positions_selected[1] == [50, 50]).all()

    def test_non_interactive_invert_selection(self):
        image = np.random.random((200, 200))
        atom_positions = [[10, 20], [50, 50]]
        verts = [(5, 5), (5, 30), (20, 30), (20, 5)]
        atom_positions_selected = ipf.select_atoms_with_gui(
            image, atom_positions, verts=verts, invert_selection=True
        )
        assert len(atom_positions_selected) == 1
        assert (atom_positions_selected[0] == [50, 50]).all()
