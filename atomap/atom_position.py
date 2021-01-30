"""This module contains the Atom_Position class.

The Atom_Position is the "base unit" in Atomap, since it contains
the information about the individual atomic columns.
"""
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
import atomap.atom_finding_refining as afr
from atomap.atom_finding_refining import _make_circular_mask, _crop_array
from atomap.atom_finding_refining import zero_array_outside_circle
from atomap.atom_finding_refining import calculate_center_of_mass
from atomap.atom_finding_refining import fit_atom_positions_gaussian
from atomap.atom_finding_refining import _atom_to_gaussian_component


class Atom_Position:
    def __init__(self, x, y, sigma_x=1.0, sigma_y=1.0, rotation=0.01, amplitude=1.0):
        """
        The Atom_Position class contain information about a single atom column.

        Parameters
        ----------
        x : float
        y : float
        sigma_x : float, optional
        sigma_y : float, optional
        rotation : float, optional
            In radians. The rotation of the axes of the 2D-Gaussian relative
            to the image axes. In other words: the rotation of the sigma_x
            relative to the horizontal axis (x-axis). This is different
            from the rotation_ellipticity, which is the rotation of the
            largest sigma in relation to the horizontal axis.
            For the rotation of the ellipticity, see rotation_ellipticity.
        amplitude : float, optional
            Amplitude of Gaussian. Stored as amplitude_gaussian attribute.

        Attributes
        ----------
        ellipticity : float
        rotation : float
            The rotation of sigma_x axis, relative to the x-axis in radians.
            This value will always be between 0 and pi, as the elliptical
            2D-Gaussian used here is always symmetric in the rotation
            direction, and the perpendicular rotation direction.
        rotation_ellipticity : float
            The rotation of the longest sigma, relative to the x-axis in
            radians.
        refine_position : bool
            If True (default), the atom position will be fitted to the image
            data when calling the
            Sublattice.refine_atom_positions_using_center_of_mass and
            Sublattice.refine_atom_positions_using_2d_gaussian methods.
            Note, the atom will still be fitted when directly calling the
            Atom_Position.refine_position_using_center_of_mass and
            Atom_Position.refine_position_using_2d_gaussian methods in the
            atom position class itself. Setting it to False can be useful when
            dealing with vacancies, or other features where the automatic
            fitting doesn't work.

        Examples
        --------
        >>> from atomap.atom_position import Atom_Position
        >>> atom_position = Atom_Position(10, 5)

        More parameters

        >>> atom_pos = Atom_Position(10, 5, sigma_x=2, sigma_y=4, rotation=2)

        """
        self.pixel_x, self.pixel_y = x, y
        self.sigma_x, self.sigma_y = sigma_x, sigma_y
        self.rotation = rotation
        self.nearest_neighbor_list = None
        self.in_atomic_plane = []
        self._start_atom = []
        self._end_atom = []
        self.atom_planes = []
        self._tag = ""
        self.old_pixel_x_list = []
        self.old_pixel_y_list = []
        self.amplitude_gaussian = amplitude
        self._gaussian_fitted = False
        self.amplitude_max_intensity = 1.0
        self.amplitude_min_intensity = 0.0
        self.intensity_mask = 0.0
        self.refine_position = True

    def __repr__(self):
        return "<%s, %s (x:%s,y:%s,sx:%s,sy:%s,r:%s,e:%s)>" % (
            self.__class__.__name__,
            self._tag,
            round(self.pixel_x, 1),
            round(self.pixel_y, 1),
            round(self.sigma_x, 1),
            round(self.sigma_y, 1),
            round(self.rotation, 1),
            round(self.ellipticity, 1),
        )

    @property
    def sigma_x(self):
        return self.__sigma_x

    @sigma_x.setter
    def sigma_x(self, new_sigma_x):
        self.__sigma_x = abs(new_sigma_x)

    @property
    def sigma_y(self):
        return self.__sigma_y

    @sigma_y.setter
    def sigma_y(self, new_sigma_y):
        self.__sigma_y = abs(new_sigma_y)

    @property
    def sigma_average(self):
        sigma = (abs(self.sigma_x) + abs(self.sigma_y)) * 0.5
        return sigma

    @property
    def rotation(self):
        """The rotation of the atom relative to the horizontal axis.

        Given in radians.
        For the rotation of the ellipticity, see rotation_ellipticity.
        """
        return self.__rotation

    @rotation.setter
    def rotation(self, new_rotation):
        self.__rotation = new_rotation % math.pi

    @property
    def rotation_ellipticity(self):
        """Rotation between the "x-axis" and the major axis of the ellipse.

        Rotation between the horizontal axis, and the longest part of the
        atom position, given by the longest sigma.
        Basically giving the direction of the ellipticity.
        """
        if self.sigma_x > self.sigma_y:
            temp_rotation = self.__rotation % math.pi
        else:
            temp_rotation = (self.__rotation + (math.pi / 2)) % math.pi
        return temp_rotation

    @property
    def ellipticity(self):
        """Largest sigma divided by the shortest"""
        if self.sigma_x > self.sigma_y:
            return self.sigma_x / self.sigma_y
        else:
            return self.sigma_y / self.sigma_x

    def as_gaussian(self):
        g = _atom_to_gaussian_component(self)
        g.A.value = self.amplitude_gaussian
        return g

    def get_pixel_position(self):
        return (self.pixel_x, self.pixel_y)

    def get_pixel_difference(self, atom):
        """Vector between self and given atom"""
        x_distance = self.pixel_x - atom.pixel_x
        y_distance = self.pixel_y - atom.pixel_y
        return (x_distance, y_distance)

    def get_angle_between_atoms(self, atom0, atom1=None):
        """
        Return the angle between atoms in radians.

        Can either find the angle between self and two other atoms,
        or between another atom and the horizontal axis.

        Parameters
        ----------
        atom0 : Atom Position object
            The first atom.
        atom1 : Atom Position object, optional
            If atom1 is not specified, the angle between
            itself, atom0 and the horizontal axis will be
            returned.

        Returns
        -------
        Angle : float
            Angle in radians

        Examples
        --------
        >>> from atomap.atom_position import Atom_Position
        >>> atom0 = Atom_Position(0, 0)
        >>> atom1 = Atom_Position(1, 1)
        >>> atom2 = Atom_Position(-1, 1)
        >>> angle0 = atom0.get_angle_between_atoms(atom1, atom2)
        >>> angle1 = atom0.get_angle_between_atoms(atom1)

        """
        vector0 = np.array([atom0.pixel_x - self.pixel_x, atom0.pixel_y - self.pixel_y])
        if atom1 is None:
            vector1 = np.array([self.pixel_x + 1000, 0])
        else:
            vector1 = np.array(
                [atom1.pixel_x - self.pixel_x, atom1.pixel_y - self.pixel_y]
            )
        cosang = np.dot(vector0, vector1)
        sinang = np.linalg.norm(np.cross(vector0, vector1))
        return np.arctan2(sinang, cosang)

    def get_angle_between_zone_vectors(self, zone_vector0, zone_vector1):
        """
        Return the angle between itself and the next atoms in
        the atom planes belonging to zone_vector0 and zone_vector1
        """
        atom0 = self.get_next_atom_in_zone_vector(zone_vector0)
        atom1 = self.get_next_atom_in_zone_vector(zone_vector1)
        if atom0 is False:
            return False
        if atom1 is False:
            return False
        angle = self.get_angle_between_atoms(atom0, atom1)
        return angle

    def _get_image_slice_around_atom(self, image_data, distance_to_edge):
        """Return a square slice of the image data.

        The atom is in the center of this slice, with the size of the image
        slice being (distance_to_edge * 2) + 1. +1 due to the position of the
        atom itself.

        If the atom is close to the edges of image_data, the size of the
        image slice will be reduced.

        Parameters
        ----------
        image_data : Numpy 2D array
        distance_to_edge : scalar
            The distance from the atom position to the borders of the image
            slice.

        Returns
        -------
        image_slice, x0, y0 : 2D numpy array

        Example
        -------
        >>> image_data = np.random.random((50, 50))
        >>> from atomap.atom_position import Atom_Position
        >>> atom = Atom_Position(10, 15)
        >>> image_slice, x0, y0 = atom._get_image_slice_around_atom(
        ...     image_data, 4)
        >>> print(image_slice.shape) # shape returns in (y, x) order
        (9, 9)

        Atom close to the edge of the image_data

        >>> image_data = np.random.random((50, 60))
        >>> atom = Atom_Position(0, 15)
        >>> image_slice, x0, y0 = atom._get_image_slice_around_atom(
        ...     image_data, 7)
        >>> print(image_slice.shape) # shape returns in (y, x) order
        (15, 8)

        """
        x = int(round(self.pixel_x))
        y = int(round(self.pixel_y))
        distance_to_edge = int(round(distance_to_edge))

        x0 = x - distance_to_edge
        x1 = x + distance_to_edge + 1
        y0 = y - distance_to_edge
        y1 = y + distance_to_edge + 1

        if x0 < 0.0:
            x0 = 0
        if y0 < 0.0:
            y0 = 0
        if x1 > image_data.shape[1]:
            x1 = image_data.shape[1]
        if y1 > image_data.shape[0]:
            y1 = image_data.shape[0]
        image_slice = copy.deepcopy(image_data[y0:y1, x0:x1])
        return image_slice, x0, y0

    def _plot_gaussian2d_debug(self, slice_radius, gaussian, data_slice):

        X, Y = np.meshgrid(
            np.arange(-slice_radius, slice_radius, 1),
            np.arange(-slice_radius, slice_radius, 1),
        )
        s_m = gaussian.function(X, Y)

        fig, axarr = plt.subplots(2, 2)
        ax0 = axarr[0][0]
        ax1 = axarr[0][1]
        ax2 = axarr[1][0]
        ax3 = axarr[1][1]

        ax0.imshow(data_slice, interpolation="nearest")
        ax1.imshow(s_m, interpolation="nearest")
        ax2.plot(data_slice.sum(0))
        ax2.plot(s_m.sum(0))
        ax3.plot(data_slice.sum(1))
        ax3.plot(s_m.sum(1))

        fig.tight_layout()
        fig.savefig(
            "debug_plot_2d_gaussian_" + str(np.random.randint(1000, 10000)) + ".jpg",
            dpi=400,
        )
        plt.close("all")

    def get_closest_neighbor(self):
        """
        Find the closest neighbor to an atom in the same sub lattice.

        Returns
        -------
        Atomap atom_position object

        """
        closest_neighbor = 100000000000000000
        for neighbor_atom in self.nearest_neighbor_list:
            distance = self.get_pixel_distance_from_another_atom(neighbor_atom)
            if distance < closest_neighbor:
                closest_neighbor = distance
        return closest_neighbor

    def calculate_max_intensity(self, image_data, percent_to_nn=0.40):
        """
        Find the maximum intensity of the atom.
        See get_atom_column_amplitude_max_intensity() for further
        uses.

        The maximum intensity is found within the distance to
        the nearest neighbor times percent_to_nn.

        Parameters
        ----------
        image_data : NumPy 2D array
        percent_to_nn : float, default 0.4
            Determines the boundary of the area surrounding each atomic
            column, as fraction of the distance to the nearest neighbour.

        Returns
        -------
        Maximum pixel intensity for an atom position.

        Example
        -------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.find_nearest_neighbors()
        >>> atom0 = sublattice.atom_list[0]
        >>> atom0_max_int = atom0.calculate_max_intensity(sublattice.image)
        """

        closest_neighbor = self.get_closest_neighbor()

        slice_size = closest_neighbor * percent_to_nn
        data_slice, x0, y0 = self._get_image_slice_around_atom(image_data, slice_size)

        data_slice_max = data_slice.max()
        self.amplitude_max_intensity = data_slice_max

        return data_slice_max

    def calculate_min_intensity(self, image_data, percent_to_nn=0.40):
        """
        Find the minimum intensity of the atom.
        See get_atom_column_amplitude_min_intensity() for further
        uses.

        The min intensity is found within the distance to
        the nearest neighbor times percent_to_nn.

        Parameters
        ----------
        image_data : NumPy 2D array
        percent_to_nn : float, default 0.4
            Determines the boundary of the area surrounding each atomic
            column, as fraction of the distance to the nearest neighbour.

        Returns
        -------
        Minimum pixel intensity for an atom position.

        Example
        -------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.find_nearest_neighbors()
        >>> atom0 = sublattice.atom_list[0]
        >>> atom0_min_int = atom0.calculate_min_intensity(sublattice.image)
        """

        closest_neighbor = self.get_closest_neighbor()

        slice_size = closest_neighbor * percent_to_nn
        data_slice, _, _ = self._get_image_slice_around_atom(image_data, slice_size)

        data_slice_min = data_slice.min()
        self.amplitude_min_intensity = data_slice_min

        return data_slice_min

    def refine_position_using_2d_gaussian(
        self,
        image_data,
        rotation_enabled=True,
        percent_to_nn=0.40,
        mask_radius=None,
        centre_free=True,
    ):
        """
        Use 2D Gaussian to refine the parameters of the atom position.

        Parameters
        ----------
        image_data : Numpy 2D array
        rotation_enabled : bool, optional
            If True, the Gaussian will be able to rotate.
            Note, this can increase the chance of fitting failure.
            Default True.
        percent_to_nn : float, optional
            The percent of the distance to the nearest neighbor atom
            in the same sub lattice. The distance times this percentage
            defines the mask around the atom where the Gaussian will be
            fitted. A smaller value can reduce the effect from
            neighboring atoms, but might also decrease the accuracy of
            the fitting due to less data to fit to.
            Default 0.4 (40%).
        mask_radius : float, optional
            Radius of the mask around each atom. If this is not set,
            the radius will be the distance to the nearest atom in the
            same sublattice times the `percent_to_nn` value.
            Note: if `mask_radius` is not specified, the Atom_Position objects
            must have a populated nearest_neighbor_list.
        centre_free : bool, default True
            If True, the centre parameter will be free, meaning that
            the Gaussian can move.

        """
        fit_atom_positions_gaussian(
            [self],
            image_data=image_data,
            rotation_enabled=rotation_enabled,
            percent_to_nn=percent_to_nn,
            mask_radius=mask_radius,
            centre_free=centre_free,
        )

    def refine_position_using_center_of_mass(
        self, image_data, percent_to_nn=0.40, mask_radius=None
    ):
        """Refine the position of the atom position using center of mass

        The position is stored in atom_position.pixel_x and
        atom_position.pixel_y. The old positions are saved in
        atom_position.old_pixel_x_list and atom_position.old_pixel_x_list.

        Parameters
        ----------
        image_data : Numpy 2D array
        percent_to_nn : float, optional
            The percent of the distance to the nearest neighbor atom
            in the same sub lattice. The distance times this percentage
            defines the mask around the atom where the Gaussian will be
            fitted. A smaller value can reduce the effect from
            neighboring atoms, but might also decrease the accuracy of
            the fitting due to less data to fit to.
            Default 0.4 (40%).
        mask_radius : float, optional
            Radius of the mask around each atom. If this is not set,
            the radius will be the distance to the nearest atom in the
            same sublattice times the `percent_to_nn` value.
            Note: if `mask_radius` is not specified, the Atom_Position objects
            must have a populated nearest_neighbor_list.

        Examples
        --------
        >>> from atomap.atom_position import Atom_Position
        >>> atom = Atom_Position(15, 10)
        >>> image_data = np.random.randint(100, size=(20, 20))
        >>> atom.refine_position_using_center_of_mass(
        ...     image_data, mask_radius=5)

        """
        new_x, new_y = self._get_center_position_com(
            image_data, percent_to_nn=percent_to_nn, mask_radius=mask_radius
        )
        self.old_pixel_x_list.append(self.pixel_x)
        self.old_pixel_y_list.append(self.pixel_y)
        self.pixel_x = new_x
        self.pixel_y = new_y

    def _get_center_position_com(
        self, image_data, percent_to_nn=0.40, mask_radius=None
    ):
        """Get new atom position based on the center of mass approach

        Parameters
        ----------
        image_data : Numpy 2D array
        percent_to_nn : float, optional
            The percent of the distance to the nearest neighbor atom
            in the same sub lattice. The distance times this percentage
            defines the mask around the atom where the Gaussian will be
            fitted. A smaller value can reduce the effect from
            neighboring atoms, but might also decrease the accuracy of
            the fitting due to less data to fit to.
            Default 0.4 (40%).
        mask_radius : float, optional
            Radius of the mask around each atom. If this is not set,
            the radius will be the distance to the nearest atom in the
            same sublattice times the `percent_to_nn` value.
            Note: if `mask_radius` is not specified, the Atom_Position objects
            must have a populated nearest_neighbor_list.

        Returns
        -------
        new_position : tuple
            (new x, new y)

        """
        if mask_radius is None:
            closest_neighbor = 100000000000000000
            for neighbor_atom in self.nearest_neighbor_list:
                distance = self.get_pixel_distance_from_another_atom(neighbor_atom)
                if distance < closest_neighbor:
                    closest_neighbor = distance
            mask_radius = closest_neighbor * percent_to_nn

        cx, cy = int(round(self.pixel_x)), int(round(self.pixel_y))
        crop_radius = np.ceil(mask_radius).astype(int)
        data = _crop_array(image_data, cx, cy, crop_radius + 1)
        edgeX, edgeY = cx - crop_radius, cy - crop_radius
        data2 = zero_array_outside_circle(data, mask_radius)
        new_y, new_x = calculate_center_of_mass(data2)
        new_x, new_y = edgeX + new_x, edgeY + new_y

        return new_x, new_y

    def get_atomic_plane_from_zone_vector(self, zone_vector):
        for atomic_plane in self.in_atomic_plane:
            if atomic_plane.zone_vector[0] == zone_vector[0]:
                if atomic_plane.zone_vector[1] == zone_vector[1]:
                    return atomic_plane
        return False

    def get_neighbor_atoms_in_atomic_plane_from_zone_vector(self, zone_vector):
        atom_plane = self.get_atomic_plane_from_zone_vector(zone_vector)
        atom_plane_atom_neighbor_list = []
        for atom in self.nearest_neighbor_list:
            if atom in atom_plane.atom_list:
                atom_plane_atom_neighbor_list.append(atom)
        return atom_plane_atom_neighbor_list

    def is_in_atomic_plane(self, zone_direction):
        for atomic_plane in self.in_atomic_plane:
            if atomic_plane.zone_vector[0] == zone_direction[0]:
                if atomic_plane.zone_vector[1] == zone_direction[1]:
                    return True
        return False

    def get_ellipticity_vector(self):
        elli = self.ellipticity - 1
        rot = self.get_ellipticity_rotation_vector()
        vector = (elli * rot[0], elli * rot[1])
        return vector

    def get_rotation_vector(self):
        rot = self.rotation
        vector = (math.cos(rot), math.sin(rot))
        return vector

    def get_ellipticity_rotation_vector(self):
        rot = self.rotation_ellipticity
        vector = (math.cos(rot), math.sin(rot))
        return vector

    def get_pixel_distance_from_another_atom(self, atom):
        x_distance = self.pixel_x - atom.pixel_x
        y_distance = self.pixel_y - atom.pixel_y
        total_distance = math.hypot(x_distance, y_distance)
        return total_distance

    def pixel_distance_from_point(self, point=(0, 0)):
        dist = math.hypot(self.pixel_x - point[0], self.pixel_y - point[1])
        return dist

    def get_index_in_atom_plane(self, atom_plane):
        for atom_index, atom in enumerate(atom_plane.atom_list):
            if atom == self:
                return atom_index

    def get_next_atom_in_atom_plane(self, atom_plane):
        current_index = self.get_index_in_atom_plane(atom_plane)
        if self == atom_plane.end_atom:
            return False
        else:
            next_atom = atom_plane.atom_list[current_index + 1]
            return next_atom

    def get_previous_atom_in_atom_plane(self, atom_plane):
        current_index = self.get_index_in_atom_plane(atom_plane)
        if self == atom_plane.start_atom:
            return False
        else:
            previous_atom = atom_plane.atom_list[current_index - 1]
            return previous_atom

    def get_next_atom_in_zone_vector(self, zone_vector):
        """Get the next atom in the atom plane belonging to zone vector."""
        atom_plane = self.get_atomic_plane_from_zone_vector(zone_vector)
        if atom_plane is False:
            return False
        next_atom = self.get_next_atom_in_atom_plane(atom_plane)
        return next_atom

    def get_previous_atom_in_zone_vector(self, zone_vector):
        atom_plane = self.get_atomic_plane_from_zone_vector(zone_vector)
        if atom_plane is False:
            return False
        previous_atom = self.get_previous_atom_in_atom_plane(atom_plane)
        return previous_atom

    def can_atom_plane_be_reached_through_zone_vector(self, atom_plane, zone_vector):
        for test_atom_plane in self.atom_planes:
            if test_atom_plane.zone_vector == zone_vector:
                for temp_atom in test_atom_plane.atom_list:
                    for temp_atom_plane in temp_atom.atom_planes:
                        if temp_atom_plane == atom_plane:
                            return test_atom_plane
        return False

    def get_position_convergence(self, distance_to_first_position=False):
        x_list = self.old_pixel_x_list
        y_list = self.old_pixel_y_list
        distance_list = []
        for index, (x, y) in enumerate(zip(x_list[1:], y_list[1:])):
            if distance_to_first_position:
                previous_x = x_list[0]
                previous_y = y_list[0]
            else:
                previous_x = x_list[index]
                previous_y = y_list[index]
            dist = math.hypot(x - previous_x, y - previous_y)
            distance_list.append(dist)
        return distance_list

    def find_atom_intensity_inside_mask(self, image_data, radius):
        """Find the average intensity inside a circle.

        The circle is defined by the atom position, and the given
        radius (in pixels).
        The outside this area is covered by a mask. The average
        intensity is saved to self.intensity_mask.
        """
        if radius is None:
            radius = 1
        centerX, centerY = self.pixel_x, self.pixel_y
        mask = _make_circular_mask(
            centerY, centerX, image_data.shape[0], image_data.shape[1], radius
        )
        data_mask = image_data * mask
        self.intensity_mask = np.mean(data_mask[np.nonzero(mask)])

    def estimate_local_scanning_distortion(self, image_data, radius=6, edge_skip=2):
        """Get the amount of local scanning distortion from an atomic column.

        This is done by assuming the atomic column has a symmetrical shape,
        like Gaussian or Lorentzian. The distortion is calculated by getting
        the image_data around the atom, given by the radius parameter.
        This gives a square cropped version of the image_data, where the
        region outside the radius is masked. For each line in the horizontal
        direction, the center of mass is found. This gives a list of
        horizontal positions as a function of the vertical lines.
        To remove the effects like astigmatism and mistilt a linear fit is
        fitted to this list of horizontal positions. This fit is then
        subtracted from the horizontal position. The distortion for the
        vertical lines is then calculated by getting the standard deviation
        of this list of values.
        Getting the horizontal distortions is calculated in a similar fashion,
        but with a list of vertical positions as a function of the horizontal
        lines.

        Parameters
        ----------
        image_data : 2D NumPy array
        radius : int
            Radius of the masked and cropped image. Default 6.
        edge_skip : int
            When the cropped image is masked with a circle,
            the edges will consist of very few unmasked pixels.
            The center of mass of these lines will be unreliable, so they're
            by default skipped. edge_skip = 2 means the two lines closest to
            the edge of the cropped image will be skipped. Default 2.

        Returns
        -------
        scanning_distortion : tuple
            Both horizontal and vertical directions. For standard raster scans,
            horizontal will be the fast scan direction,
            and y the slow scan direction. Thus the returned values will be:
            (fast scan, slow scan), where typically the slow scan direction has
            the largest amount of distortion.

        Examples
        --------
        >>> sl = am.dummy_data.get_scanning_distortion_sublattice()
        >>> atom = sl.atom_list[50]
        >>> distortion = atom.estimate_local_scanning_distortion(sl.image)

        """
        atom_image = self._get_image_slice_around_atom(image_data, radius)[0]
        atom_mask = afr._make_mask_circle_centre(atom_image, radius)
        atom_image[atom_mask] = 0

        line_x_com_list = []
        for ix in range(edge_skip, atom_image.shape[1] - edge_skip):
            line_mask_x = atom_mask[:, ix]
            com_x_offset = line_mask_x[: round(len(line_mask_x) / 2)].sum()
            line_x = atom_image[:, ix][np.invert(line_mask_x)]
            if np.any(line_x):
                line_x_com = center_of_mass(line_x)[0] + com_x_offset
                line_x_com_list.append(line_x_com)

        line_x_com_range = range(len(line_x_com_list))
        line_x_com_poly = np.polyfit(line_x_com_range, line_x_com_list, deg=1)
        line_x_com_fit = np.poly1d(line_x_com_poly)(line_x_com_range)
        line_x_variation = np.array(line_x_com_list) - np.array(line_x_com_fit)

        line_y_com_list = []
        for iy in range(edge_skip, atom_image.shape[0] - edge_skip):
            line_mask_y = atom_mask[iy]
            com_y_offset = line_mask_y[: round(len(line_mask_y) / 2)].sum()
            line_y = atom_image[iy][np.invert(line_mask_y)]
            if np.any(line_y):
                line_y_com = center_of_mass(line_y)[0] + com_y_offset
                line_y_com_list.append(line_y_com)

        line_y_com_range = range(len(line_y_com_list))
        line_y_com_poly = np.polyfit(line_y_com_range, line_y_com_list, deg=1)
        line_y_com_fit = np.poly1d(line_y_com_poly)(line_y_com_range)
        line_y_variation = np.array(line_y_com_list) - np.array(line_y_com_fit)

        line_x_std = np.std(line_x_variation)
        line_y_std = np.std(line_y_variation)
        return line_x_std, line_y_std

    def _get_atom_slice(self, im_x, im_y, sigma_quantile=5):
        """Get a 2D slice for slicing an image, based on the centre and sigma

        slice is defined by:
        x - sigma_x * sigma_quantile, x + sigma_x * sigma_quantile
        y - sigma_y * sigma_quantile, y + sigma_y * sigma_quantile

        Parameters
        ----------
        im_x, im_y : int
            x and y size of the image.
        sigma_quantile : scalar

        Returns
        -------
        atom_slice : tuple of slices

        Examples
        --------
        >>> from atomap.atom_position import Atom_Position
        >>> atom = Atom_Position(x=15, y=10, sigma_x=5, sigma_y=3)
        >>> atom_slice = atom._get_atom_slice(100, 150, sigma_quantile=4)

        """
        x, y = self.pixel_x, self.pixel_y
        smax = max(self.sigma_x, self.sigma_y)
        ix0 = math.floor(x - (smax * sigma_quantile))
        ix1 = math.ceil(x + (smax * sigma_quantile))
        iy0 = math.floor(y - (smax * sigma_quantile))
        iy1 = math.ceil(y + (smax * sigma_quantile))
        ix0, iy0 = max(0, ix0), max(0, iy0)
        ix1, iy1 = min(im_x, ix1), min(im_y, iy1)
        atom_slice = np.s_[iy0:iy1, ix0:ix1]
        return atom_slice

    def set_element_info(self, element, z):
        """Set which atoms are present along the atomic column at this atom
        position.

        Parameters
        ----------
        element : str or list of str
            elements contained in the atomic column.
        z : list of floats
            In Ångstrøm

        Examples
        --------
        >>> from atomap.atom_position import Atom_Position
        >>> atom = Atom_Position(x=15, y=10, sigma_x=5, sigma_y=3)
        >>> atom.set_element_info("C", [0, 2.5])
        >>> atom2 = Atom_Position(x=10, y=15, sigma_x=5, sigma_y=3)
        >>> atom2.set_element_info(["C", "O"], [0, 2.5])
        """

        self.element_info = {}

        if isinstance(element, str):
            for zc in z:
                self.element_info[zc] = element
        elif isinstance(element, list):
            for i, zc in enumerate(z):
                self.element_info[zc] = element[i]
        else:
            raise TypeError("element must be either a str or list of str")
