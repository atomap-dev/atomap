import numpy as np
import operator
import copy
from scipy.stats import linregress
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from atomap.fitting_tools import (
    ODR_linear_fitter,
    linear_fit_func,
    get_shortest_distance_point_to_line,
)


class Atom_Plane:
    def __init__(self, atom_list, zone_vector, atom_lattice):
        """
        Parameters
        ----------
        atom_list : list of Atom_Position objects
        zone_vector : tuple
        atom_lattice : Atomap Atom_Lattice object

        Attributes
        ----------
        x_position : list of floats
        y_position : list of floats
        sigma_x : list of floats
        sigma_y : list of floats
        sigma_average : list of floats
        rotation : list of floats
        ellipticity : list of floats
        """
        self.atom_list = atom_list
        self.zone_vector = zone_vector
        self.atom_lattice = atom_lattice
        self.start_atom = None
        self.end_atom = None
        self._find_start_atom()
        self._find_end_atom()
        self.sort_atoms_by_distance_to_point(self.start_atom.get_pixel_position())

        self.atom_distance_list = self.get_atom_distance_list()
        self._link_atom_to_atom_plane()

    def __repr__(self):
        return "<%s, %s (atoms:%s)>" % (
            self.__class__.__name__,
            self.zone_vector,
            len(self.atom_list),
        )

    @property
    def x_position(self):
        x_pos = []
        for atom in self.atom_list:
            x_pos.append(atom.pixel_x)
        return x_pos

    @property
    def y_position(self):
        y_pos = []
        for atom in self.atom_list:
            y_pos.append(atom.pixel_y)
        return y_pos

    @property
    def sigma_x(self):
        sigma_x = []
        for atom in self.atom_list:
            sigma_x.append(abs(atom.sigma_x))
        return sigma_x

    @property
    def sigma_y(self):
        sigma_y = []
        for atom in self.atom_list:
            sigma_y.append(abs(atom.sigma_y))
        return sigma_y

    @property
    def sigma_average(self):
        sigma = np.array(self.sigma_x) + np.array(self.sigma_y)
        sigma *= 0.5
        return sigma.tolist()

    @property
    def amplitude_gaussian(self):
        amplitude = []
        for atom in self.atom_list:
            amplitude.append(atom.amplitude_gaussian)
        return amplitude

    @property
    def amplitude_max_intensity(self):
        amplitude = []
        for atom in self.atom_list:
            amplitude.append(atom.amplitude_max_intensity)
        return amplitude

    @property
    def rotation(self):
        rotation = []
        for atom in self.atom_list:
            rotation.append(atom.rotation)
        return rotation

    @property
    def ellipticity(self):
        ellipticity = []
        for atom in self.atom_list:
            ellipticity.append(atom.ellipticity)
        return ellipticity

    @property
    def rotation_ellipticity(self):
        rotation_ellipticity = []
        for atom in self.atom_list:
            rotation_ellipticity.append(atom.rotation_ellipticity)
        return rotation_ellipticity

    def _link_atom_to_atom_plane(self):
        for atom in self.atom_list:
            atom.atom_planes.append(self)

    def get_x_position_list(self):
        x_position_list = []
        for atom in self.atom_list:
            x_position_list.append(atom.pixel_x)
        return x_position_list

    def get_y_position_list(self):
        y_position_list = []
        for atom in self.atom_list:
            y_position_list.append(atom.pixel_y)
        return y_position_list

    def _find_start_atom(self):
        for atom in self.atom_list:
            if self.zone_vector in atom._start_atom:
                self.start_atom = atom
                break

    def _find_end_atom(self):
        for atom in self.atom_list:
            if self.zone_vector in atom._end_atom:
                self.end_atom = atom
                break

    def get_intersecting_atom_from_atom_plane(self, atom_plane):
        for self_atom in self.atom_list:
            if self_atom in atom_plane.atom_list:
                return self_atom
        return "Intersecting atom not found"

    def sort_atoms_by_distance_to_point(self, point=(0, 0)):
        self.atom_list.sort(
            key=operator.methodcaller("pixel_distance_from_point", point)
        )

    def get_slice_between_two_atoms(self, atom1, atom2):
        if not (atom1 in self.atom_list) and not (atom2 in self.atom_list):
            return False
        atom1_is_first = None
        for atom in self.atom_list:
            if atom == atom1:
                atom1_is_first = True
                break
            elif atom == atom2:
                atom1_is_first = False
                break
        atom_list = []
        if atom1_is_first:
            while not (atom1 == self.end_atom):
                atom_list.append(atom1)
                atom1 = atom1.get_next_atom_in_atom_plane(self)
                if atom1 == atom2:
                    atom_list.append(atom2)
                    break
        return atom_list

    def get_atom_distance_list(self):
        atom_distances = []
        for atom_index, atom in enumerate(self.atom_list):
            if not (atom_index == 0):
                distance = atom.get_pixel_distance_from_another_atom(
                    self.atom_list[atom_index - 1]
                )
                atom_distances.append(distance)
        return atom_distances

    def position_distance_to_neighbor(self):
        """
        Get distance between all atoms and its next neighbor
        in the atom plane, and the position between these two atoms.

        Returns
        -------
        Numpy array [x, y, distance]

        Example
        -------
        >>> from numpy.random import random
        >>> from atomap.sublattice import Sublattice
        >>> pos = [[x, y] for x in range(9) for y in range(9)]
        >>> sublattice = Sublattice(pos, random((9, 9)))
        >>> sublattice.construct_zone_axes()
        >>> atom_plane = sublattice.atom_plane_list[10]
        >>> pos_distance = atom_plane.position_distance_to_neighbor()
        >>> x_pos = pos_distance[:, 0]
        >>> y_pos = pos_distance[:, 1]
        >>> distance = pos_distance[:, 2]
        """
        atom_distances = []
        if len(self.atom_list) < 2:
            return None
        for atom_index, atom in enumerate(self.atom_list):
            if not (atom_index == 0):
                previous_atom = self.atom_list[atom_index - 1]
                difference_vector = previous_atom.get_pixel_difference(atom)
                pixel_x = previous_atom.pixel_x - difference_vector[0] / 2
                pixel_y = previous_atom.pixel_y - difference_vector[1] / 2
                distance = atom.get_pixel_distance_from_another_atom(previous_atom)
                atom_distances.append([pixel_x, pixel_y, distance])
        atom_distances = np.array(atom_distances)
        return atom_distances

    def get_connecting_atom_planes(self, atom_plane, zone_vector):
        """
        Get the outer atom planes which connect self atom
        plane with another atom plane, through a specific
        atom plane direction.

        The self atom plane, atom plane, and the two returned
        atom planes will span make a parallelogram.

        Parameters
        ----------
        atom_plane : Atomap atom_plane object
        zone_vector : tuple

        Returns
        -------
        tuple, two atom plane objects
        """
        start_orthogonal_atom_plane = None
        atom = self.start_atom
        while start_orthogonal_atom_plane is None:
            temp_plane = atom.can_atom_plane_be_reached_through_zone_vector(
                atom_plane, zone_vector
            )
            if temp_plane is False:
                atom = atom.get_next_atom_in_atom_plane(self)
                if atom is False:
                    break
            else:
                start_orthogonal_atom_plane = temp_plane

        end_orthogonal_atom_plane = None
        atom = self.end_atom
        while end_orthogonal_atom_plane is None:
            temp_plane = atom.can_atom_plane_be_reached_through_zone_vector(
                atom_plane, zone_vector
            )
            if temp_plane is False:
                atom = atom.get_previous_atom_in_atom_plane(self)
                if atom is False:
                    break
            else:
                end_orthogonal_atom_plane = temp_plane
        return (start_orthogonal_atom_plane, end_orthogonal_atom_plane)

    def get_net_distance_change_between_atoms(self):
        """Output [(x,y,z)]"""
        if len(self.atom_list) < 3:
            return None
        data = self.position_distance_to_neighbor()
        data = np.array(data)
        x_pos_list = data[:, 0]
        y_pos_list = data[:, 1]
        z_pos_list = data[:, 2]
        new_data_list = []
        for index, (x_pos, y_pos, z_pos) in enumerate(
            zip(x_pos_list, y_pos_list, z_pos_list)
        ):
            if not (index == 0):
                previous_x_pos = x_pos_list[index - 1]
                previous_y_pos = y_pos_list[index - 1]
                previous_z_pos = z_pos_list[index - 1]

                new_x_pos = (x_pos + previous_x_pos) * 0.5
                new_y_pos = (y_pos + previous_y_pos) * 0.5
                new_z_pos = z_pos - previous_z_pos
                new_data_list.append([new_x_pos, new_y_pos, new_z_pos])
        new_data_list = np.array(new_data_list)
        return new_data_list

    def get_atom_index(self, check_atom):
        for atom_index, atom in enumerate(self.atom_list):
            if atom == check_atom:
                return atom_index

    def get_closest_position_to_point(self, point_position, extend_line=False):
        x_pos = self.get_x_position_list()
        y_pos = self.get_y_position_list()

        if (max(x_pos) - min(x_pos)) > (max(y_pos) - min(y_pos)):
            pos_list0 = copy.deepcopy(x_pos)
            pos_list1 = copy.deepcopy(y_pos)
        else:
            pos_list0 = copy.deepcopy(y_pos)
            pos_list1 = copy.deepcopy(x_pos)

        if extend_line:
            reg_results = linregress(pos_list0[:4], pos_list1[:4])
            delta_0 = (
                np.mean((np.array(pos_list0[0:3]) - np.array(pos_list0[1:4]).mean()))
                * 40
            )
            delta_1 = reg_results[0] * delta_0
            start_0 = delta_0 + pos_list0[0]
            start_1 = delta_1 + pos_list1[0]
            pos_list0.insert(0, start_0)
            pos_list1.insert(0, start_1)

            reg_results = linregress(pos_list0[-4:], pos_list1[-4:])
            delta_0 = (
                np.mean((np.array(pos_list0[-3:]) - np.array(pos_list0[-4:-1]).mean()))
                * 40
            )
            delta_1 = reg_results[0] * delta_0
            end_0 = delta_0 + pos_list0[-1]
            end_1 = delta_1 + pos_list1[-1]
            pos_list0.append(end_0)
            pos_list1.append(end_1)

        f = interpolate.interp1d(pos_list0, pos_list1)

        new_pos_list0 = np.linspace(pos_list0[0], pos_list0[-1], len(pos_list0) * 100)
        new_pos_list1 = f(new_pos_list0)

        if (max(x_pos) - min(x_pos)) > (max(y_pos) - min(y_pos)):
            new_x = new_pos_list0
            new_y = new_pos_list1
        else:
            new_y = new_pos_list0
            new_x = new_pos_list1

        x_position_point = point_position[0]
        y_position_point = point_position[1]

        dist_x = new_x - x_position_point
        dist_y = new_y - y_position_point

        distance = (dist_x ** 2 + dist_y ** 2) ** 0.5

        closest_index = distance.argmin()
        closest_point = (new_x[closest_index], new_y[closest_index])
        return closest_point

    def get_closest_distance_and_angle_to_point(
        self, points_x, points_y, use_precalculated_line=False, plot_debug=False
    ):
        """Return the smallest distances from each point in a list to
        the atom plane.

        Parameters
        ----------
        points_x, points_y : list of numbers
            The x and y coordinates for the atoms.
        use_precalculated_line : bool or list/array
            The coefficients [a, b] for the linear line y = ax + b to which
            the shortest distance should be found. By default False, in which
            the coefficients are found by fitting a straight line to
            self (atom plane).
        plot_debug : bool, default False
            If true, a debug plot is saved. The plot is a 3D plot of x and
            y positions and distance to the plane.

        Returns
        -------
        list of numbers : list of the shortest distances.

        """
        if use_precalculated_line is False:
            x_pos = self.get_x_position_list()
            y_pos = self.get_y_position_list()
            beta = ODR_linear_fitter(x_pos, y_pos)
        else:
            beta = use_precalculated_line

        dist = get_shortest_distance_point_to_line(points_x, points_y, beta)

        if plot_debug:
            x_pos = self.get_x_position_list()
            y_pos = self.get_y_position_list()
            x0, x1 = min(x_pos), max(x_pos)
            new_x = np.linspace(x0, x1, 200)
            new_y = linear_fit_func(beta, new_x)
            plt.ioff()
            fig = plt.figure()
            ax = fig.gca(projection=Axes3D.name)
            ax.plot(new_x, new_y, color="r", lw=1)
            ax.scatter(x_pos, y_pos, color="r")
            ax.scatter(points_x, points_y, dist, s=1)
            ax.set_zlabel("Distance to plane")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.view_init(elev=0, azim=10)
            fig.savefig(str(np.random.randint(1000, 20000)) + ".png")
            plt.close()

        return dist

    def _plot_debug_atom_plane(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(self.atom_lattice.image)
        if self.atom_lattice._plot_clim:
            clim = self.atom_lattice._plot_clim
            cax.set_clim(clim[0], clim[1])
        for atom_index, atom in enumerate(self.atom_list):
            ax.plot(atom.pixel_x, atom.pixel_y, "o", color="blue")
            ax.text(atom.pixel_x, atom.pixel_y, str(atom_index))
        ax.set_ylim(0, self.atom_lattice.image.shape[0])
        ax.set_xlim(0, self.atom_lattice.image.shape[1])
        fig.tight_layout()
        fig.savefig("debug_plot_atom_plane.jpg")

    def get_angle_to_horizontal_axis(self):
        """Get angle between atoms in the atom plane and horizontal
        axis."""
        angle_list = []
        atom = self.start_atom
        while atom is not False:
            next_atom = atom.get_next_atom_in_atom_plane(self)
            if next_atom is False:
                break
            angle = atom.get_angle_between_atoms(next_atom)
            angle_list.append(angle)
            atom = next_atom
        return angle_list
