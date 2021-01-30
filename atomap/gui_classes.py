import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import atomap.tools as to


class AtomToggleRefine:
    def __init__(self, image, sublattice, distance_threshold=4):
        self.image = image
        self.distance_threshold = distance_threshold
        self.sublattice = sublattice
        self.fig, self.ax = plt.subplots()
        self.cax = self.ax.imshow(self.image)
        x_pos = self.sublattice.x_position
        y_pos = self.sublattice.y_position
        refine_list = self._get_refine_position_list(self.sublattice.atom_list)
        color_list = self._refine_position_list_to_color_list(refine_list)
        self.path = self.ax.scatter(x_pos, y_pos, c=color_list)
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.fig.tight_layout()

    def _get_refine_position_list(self, atom_list):
        refine_position_list = []
        for atom in atom_list:
            refine_position_list.append(atom.refine_position)
        return refine_position_list

    def _refine_position_list_to_color_list(
        self, refine_position_list, color_true="green", color_false="red"
    ):
        color_list = []
        for refine_position in refine_position_list:
            if refine_position:
                color_list.append(color_true)
            else:
                color_list.append(color_false)
        return color_list

    def onclick(self, event):
        if event.inaxes != self.ax.axes:
            return
        if event.button == 1:  # Left mouse button
            x = np.float(event.xdata)
            y = np.float(event.ydata)
            atom_nearby = self.is_atom_nearby(x, y)
            if atom_nearby is not None:
                ref_pos_current = self.sublattice.atom_list[atom_nearby].refine_position
                self.sublattice.atom_list[
                    atom_nearby
                ].refine_position = not ref_pos_current
                self.replot()

    def is_atom_nearby(self, x_press, y_press):
        dt = self.distance_threshold
        index = None
        closest_dist = 9999999999999999
        x_pos = self.sublattice.x_position
        y_pos = self.sublattice.y_position
        for i, (x, y) in enumerate(zip(x_pos, y_pos)):
            if x - dt < x_press < x + dt:
                if y - dt < y_press < y + dt:
                    dist = math.hypot(x_press - x, y_press - y)
                    if dist < closest_dist:
                        index = i
        return index

    def replot(self):
        refine_list = self._get_refine_position_list(self.sublattice.atom_list)
        color_list = self._refine_position_list_to_color_list(refine_list)
        self.path.set_color(color_list)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class GetAtomSelection:
    def __init__(self, image, atom_positions, invert_selection=False):
        """Get a subset of atom positions using interactive tool.

        Access the selected atom positions in the
        atom_positions_selected attribute.

        Parameters
        ----------
        image : 2D HyperSpy signal or 2D NumPy array
        atom_positions : list of lists, NumPy array
            In the form [[x0, y0]. [x1, y1], ...]
        invert_selection : bool, optional
            Get the atom positions outside the region, instead of the
            ones inside it. Default False

        Attributes
        ----------
        atom_positions_selected : NumPy array

        """
        self.image = image
        self.atom_positions = np.array(atom_positions)
        self.invert_selection = invert_selection
        self.atom_positions_selected = []
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title(
            "Use the left mouse button to make the polygon.\n"
            "Click the first position to finish the polygon.\n"
            "Press ESC to reset the polygon, and hold SHIFT to\n"
            "move the polygon."
        )
        self.cax = self.ax.imshow(self.image)
        self.line_non_selected = self.ax.plot(
            self.atom_positions[:, 0], self.atom_positions[:, 1], "o", color="red"
        )[0]
        self.line_selected = None
        markerprops = dict(color="blue")
        lineprops = dict(color="blue")
        self.poly = PolygonSelector(
            self.ax, self.onselect, markerprops=markerprops, lineprops=lineprops
        )
        self.fig.tight_layout()

    def onselect(self, verts):
        atom_positions_selected = to._get_atom_selection_from_verts(
            self.atom_positions, verts, invert_selection=self.invert_selection
        )
        atom_positions_not_selected = to._get_atom_selection_from_verts(
            self.atom_positions, verts, invert_selection=not self.invert_selection
        )
        if len(atom_positions_selected) != 0:
            if self.line_selected is None:
                self.line_selected = self.ax.plot(
                    atom_positions_selected[:, 0],
                    atom_positions_selected[:, 1],
                    "o",
                    color="green",
                )[0]
            else:
                self.line_selected.set_data(
                    atom_positions_not_selected[:, 0], atom_positions_not_selected[:, 1]
                )
            self.line_selected.set_data(
                atom_positions_selected[:, 0], atom_positions_selected[:, 1]
            )

        for atom_positions in atom_positions_selected:
            self.atom_positions_selected.append(atom_positions.tolist())
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
