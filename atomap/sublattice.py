import numpy as np
import scipy as sp
from hyperspy.external.progressbar import progressbar
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import hyperspy.api as hs
from hyperspy.signals import Signal2D
from hyperspy.drawing._markers.point import Point

import atomap.tools as at
import atomap.analysis_tools as an
from atomap.plotting import (
    _make_atom_planes_marker_list,
    _make_atom_position_marker_list,
    _make_arrow_marker_list,
    _make_multidim_atom_plane_marker_list,
    _make_zone_vector_text_marker_list,
    plot_vector_field,
    vector_list_to_marker_list,
)
from atomap.atom_finding_refining import construct_zone_axes_from_sublattice
from atomap.atom_finding_refining import _make_circular_mask

from atomap.atom_position import Atom_Position
from atomap.atom_plane import Atom_Plane
from atomap.symmetry_finding import _remove_parallel_vectors
import atomap.gui_classes as gui

from atomap.external.add_marker import add_marker
from atomap.external.gaussian2d import Gaussian2D


class Sublattice:
    def __init__(
        self,
        atom_position_list,
        image,
        original_image=None,
        name="",
        color="red",
        pixel_size=None,
        units=None,
    ):
        """
        Parameters
        ----------
        atom_position_list : NumPy array
            In the form [[x0, y0], [x1, y1], [x2, y2], ... ]
        image : HyperSpy signal, 2D NumPy array or 2D array-like
            A HyperSpy signal with 2 dimensions can also be used directly.
            Pixel size and units will be copied from a HyperSpy signal
        original_image : 2D NumPy array, optional
        name : string, default ''
        color : string, optional
            Plotting color, default red.
        pixel_size : float, optional
            Scaling number, default 1.
        units : string, optional
            Default "pixel".

        Attributes
        ----------
        image: 2D NumPy array
        x_position : NumPy Array
        y_position : NumPy Array
        atom_positions : NumPy Array
            In the form [[x0, y0], [x1, y1], ...]
        sigma_x : NumPy Array
        sigma_y : NumPy Array
        sigma_average : NumPy Array
        rotation : NumPy Array
            In radians. The rotation of the axes of each 2D-Gaussian relative
            to the horizontal axes. For the rotation of the ellipticity, see
            rotation_ellipticity.
        ellipticity : NumPy Array
        rotation_ellipticity : NumPy Array
            In radians, the rotation between the "x-axis" and the major axis
            of the ellipse. Basically giving the direction of the ellipticity.
        signal : HyperSpy 2D signal
        name : string

        Examples
        --------
        >>> import numpy as np
        >>> import atomap.api as am
        >>> atom_positions = [[2, 2], [2, 4], [4, 2], [4, 4]]
        >>> image_data = np.random.random((7, 7))
        >>> sublattice = am.Sublattice(atom_positions, image_data)
        >>> s_sublattice = sublattice.get_atom_list_on_image()
        >>> s_sublattice.plot()

        Add pixel size and units when creating sublattice

        >>> sublattice = am.Sublattice(
        ...     atom_positions, image_data, pixel_size=5, units='nm')
        >>> sublattice.plot()

        Using a HyperSpy signal2D as image. Here, the pixel_size and units
        will be taken from the signal. So if the signal is calibrated,
        the sublattice will be as well.

        >>> s = am.dummy_data.get_fantasite()
        >>> atom_positions = am.get_atom_positions(s)
        >>> sublattice = am.Sublattice(atom_positions, s)

        More atom positions

        >>> x, y = np.mgrid[0:100:10j, 0:100:10j]
        >>> x, y = x.flatten(), y.flatten()
        >>> atom_positions = np.dstack((x, y))[0]
        >>> image_data = np.random.random((100, 100))
        >>> sublattice = am.Sublattice(
        ...     atom_positions, image_data, color='yellow',
        ...     name='the sublattice')
        >>> sublattice.get_atom_list_on_image(markersize=50).plot()

        """
        at._image_init(
            self,
            image=image,
            original_image=original_image,
            pixel_size=pixel_size,
            units=units,
        )
        self.atom_list = []
        for atom_position in atom_position_list:
            atom = Atom_Position(atom_position[0], atom_position[1])
            self.atom_list.append(atom)
        self.zones_axis_average_distances = None
        self.zones_axis_average_distances_names = []
        self.atom_plane_list = []
        self.atom_planes_by_zone_vector = {}
        self._plot_clim = None
        self.name = name
        self._plot_color = color
        self._pixel_separation = 0.0

    def __repr__(self):
        return "<%s, %s (atoms:%s,planes:%s)>" % (
            self.__class__.__name__,
            self.name,
            len(self.atom_list),
            len(self.atom_planes_by_zone_vector),
        )

    @property
    def atom_positions(self):
        atom_pos = np.stack((self.x_position, self.y_position), axis=-1)
        return atom_pos

    @property
    def x_position(self):
        x_pos = []
        for atom in self.atom_list:
            x_pos.append(atom.pixel_x)
        x_pos = np.array(x_pos)
        return x_pos

    @property
    def y_position(self):
        y_pos = []
        for atom in self.atom_list:
            y_pos.append(atom.pixel_y)
        y_pos = np.array(y_pos)
        return y_pos

    @property
    def sigma_x(self):
        sigma_x = []
        for atom in self.atom_list:
            sigma_x.append(abs(atom.sigma_x))
        sigma_x = np.array(sigma_x)
        return sigma_x

    @property
    def sigma_y(self):
        sigma_y = []
        for atom in self.atom_list:
            sigma_y.append(abs(atom.sigma_y))
        sigma_y = np.array(sigma_y)
        return sigma_y

    @property
    def sigma_average(self):
        sigma = self.sigma_x + self.sigma_y
        sigma *= 0.5
        return sigma

    @property
    def atom_amplitude_gaussian2d(self):
        amplitude = []
        for atom in self.atom_list:
            amplitude.append(atom.amplitude_gaussian)
        amplitude = np.array(amplitude)
        return amplitude

    @property
    def atom_amplitude_max_intensity(self):
        amplitude = []
        for atom in self.atom_list:
            amplitude.append(atom.amplitude_max_intensity)
        amplitude = np.array(amplitude)
        return amplitude

    @property
    def atom_amplitude_min_intensity(self):
        amplitude = []
        for atom in self.atom_list:
            amplitude.append(atom.amplitude_min_intensity)
        amplitude = np.array(amplitude)
        return amplitude

    @property
    def rotation(self):
        rotation = []
        for atom in self.atom_list:
            rotation.append(atom.rotation)
        rotation = np.array(rotation)
        return rotation

    @property
    def ellipticity(self):
        ellipticity = []
        for atom in self.atom_list:
            ellipticity.append(atom.ellipticity)
        ellipticity = np.array(ellipticity)
        return ellipticity

    @property
    def rotation_ellipticity(self):
        rotation_ellipticity = []
        for atom in self.atom_list:
            rotation_ellipticity.append(atom.rotation_ellipticity)
        rotation_ellipticity = np.array(rotation_ellipticity)
        return rotation_ellipticity

    @property
    def intensity_mask(self):
        intensity_mask = []
        for atom in self.atom_list:
            intensity_mask.append(atom.intensity_mask)
        intensity_mask = np.array(intensity_mask)
        return intensity_mask

    @property
    def signal(self):
        s = Signal2D(self.image)
        s.axes_manager.signal_axes[0].scale = self.pixel_size
        s.axes_manager.signal_axes[1].scale = self.pixel_size
        s.axes_manager.signal_axes[0].units = self.units
        s.axes_manager.signal_axes[1].units = self.units
        return s

    def get_zone_vector_index(self, zone_vector_id):
        """Find zone vector index from zone vector name"""
        for zone_vector_index, zone_vector in enumerate(
            self.zones_axis_average_distances_names
        ):
            if zone_vector == zone_vector_id:
                return zone_vector_index
        raise ValueError("Could not find zone_vector " + str(zone_vector_id))

    def _check_if_nearest_neighbor_list(self):
        """Check if the nearest neighbor lists has been populated."""
        if self.atom_list[0].nearest_neighbor_list is None:
            raise Exception(
                "The atom_position objects does not seem to have a "
                "populated nearest neighbor list. "
                "Has sublattice.find_nearest_neighbors() been called?"
            )

    def _check_if_zone_axis_list(self):
        """Check if the zone axis list has been populated."""
        if self.zones_axis_average_distances is None:
            raise Exception(
                "zones_axis_average_distances is empty. "
                "Has sublattice.construct_zone_axes() been called?"
            )
        elif len(self.zones_axis_average_distances) == 0:
            raise Exception(
                "zones_axis_average_distances is empty. "
                "Has sublattice.construct_zone_axes() been called?"
            )

    def get_atom_angles_from_zone_vector(
        self, zone_vector0, zone_vector1, degrees=False
    ):
        """
        Calculates for each atom in the sub lattice the angle
        between the atom, and the next atom in the atom planes
        in zone_vector0 and zone_vector1.
        Default will return the angles in radians.

        Parameters
        ----------
        zone_vector0 : tuple
            Vector for the first zone.
        zone_vector1 : tuple
            Vector for the second zone.
        degrees : bool, optional
            If True, will return the angles in degrees.
            Default False.
        """
        angle_list = []
        pos_x_list = []
        pos_y_list = []
        for atom in self.atom_list:
            angle = atom.get_angle_between_zone_vectors(zone_vector0, zone_vector1)
            if angle is not False:
                angle_list.append(angle)
                pos_x_list.append(atom.pixel_x)
                pos_y_list.append(atom.pixel_y)
        angle_list = np.array(angle_list)
        pos_x_list = np.array(pos_x_list)
        pos_y_list = np.array(pos_y_list)
        if degrees:
            angle_list = np.rad2deg(angle_list)

        return (pos_x_list, pos_y_list, angle_list)

    def get_atom_distance_list_from_zone_vector(self, zone_vector):
        """
        Get distance between each atom and the next atom in an
        atom plane given by the zone_vector. Returns the x- and
        y-position, and the distance between the atom and the
        monolayer. The position is set between the atom and the
        monolayer.

        For certain zone axes where there is several monolayers
        between the atoms in the atom plane, there will be some
        artifacts created. For example, in the perovskite (110)
        projection, the atoms in the <111> atom planes are
        separated by 3 monolayers.

        To avoid this problem use the function
        get_monolayer_distance_list_from_zone_vector.

        Parameters
        ----------
        zone_vector : tuple
            Zone vector for the system
        """
        atom_plane_list = self.atom_planes_by_zone_vector[zone_vector]
        atom_distance_list = []
        for atom_plane in atom_plane_list:
            dist = atom_plane.position_distance_to_neighbor()
            atom_distance_list.extend(dist)
        atom_distance_list = np.array(atom_distance_list).swapaxes(0, 1)
        return (atom_distance_list[0], atom_distance_list[1], atom_distance_list[2])

    def get_monolayer_distance_list_from_zone_vector(self, zone_vector):
        """
        Get distance between each atom and the next monolayer given
        by the zone_vector. Returns the x- and y-position, and the
        distance between the atom and the monolayer. The position
        is set between the atom and the monolayer.

        The reason for finding the distance between monolayer,
        instead of directly between the atoms is due to some zone axis
        having having a rather large distance between the atoms in
        one atom plane. For example, in the perovskite (110) projection,
        the atoms in the <111> atom planes are separated by 3 monolayers.
        This can give very bad results.

        To get the distance between atoms, use the function
        get_atom_distance_list_from_zone_vector.

        Parameters
        ----------
        zone_vector : tuple
            Zone vector for the system
        """
        self._check_if_zone_axis_list()
        atom_plane_list = self.atom_planes_by_zone_vector[zone_vector]
        x_list, y_list, z_list = [], [], []
        for index, atom_plane in enumerate(atom_plane_list[1:]):
            atom_plane_previous = atom_plane_list[index]
            plane_data_list = self._get_distance_and_position_list_between_atom_planes(
                atom_plane_previous, atom_plane
            )
            x_list.extend(plane_data_list[0].tolist())
            y_list.extend(plane_data_list[1].tolist())
            z_list.extend(plane_data_list[2].tolist())
        return (x_list, y_list, z_list)

    def get_atom_distance_difference_from_zone_vector(self, zone_vector):
        """
        Get distance difference between atoms in atoms planes
        belonging to a zone axis.

        Parameters
        ----------
        zone_vector : tuple
            Zone vector for the system
        """
        x_list, y_list, z_list = [], [], []
        for atom_plane in self.atom_planes_by_zone_vector[zone_vector]:
            data = atom_plane.get_net_distance_change_between_atoms()
            if data is not None:
                x_list.extend(data[:, 0])
                y_list.extend(data[:, 1])
                z_list.extend(data[:, 2])
        return (x_list, y_list, z_list)

    def _property_position(self, property_list, x_position=None, y_position=None):
        if x_position is None:
            x_position = self.x_position
        if y_position is None:
            y_position = self.y_position
        data_list = np.array([x_position, y_position, property_list])
        data_list = np.swapaxes(data_list, 0, 1)
        return data_list

    def _property_position_projection(
        self,
        interface_plane,
        property_list,
        x_position=None,
        y_position=None,
        scale_xy=1.0,
        scale_z=1.0,
    ):
        if x_position is None:
            x_position = self.x_position
        if y_position is None:
            y_position = self.y_position
        x, y = np.array(x_position), np.array(y_position)
        z = np.array(property_list)
        xs, ys, zs = x.shape, y.shape, z.shape
        if (xs != ys) or (xs != zs) or (ys != zs):
            raise ValueError(
                "x_position, y_position and property_list must have the " "same shape"
            )
        data_list = np.array([x, y, z])
        data_list = np.swapaxes(data_list, 0, 1)
        projected_positions = at.project_position_property(data_list, interface_plane)
        layer_list = at.sort_projected_positions_into_layers(projected_positions)
        line_profile_data = at.combine_projected_positions_layers(layer_list)
        line_profile_data = np.array(line_profile_data)
        position = line_profile_data[:, 0] * scale_xy
        data = line_profile_data[:, 1] * scale_z
        std = line_profile_data[:, 2] * scale_z
        return np.array([position, data, std])

    def _get_regular_grid_from_unregular_property(
        self, x_list, y_list, z_list, upscale=2
    ):
        """
        Interpolate irregularly spaced data points into a
        regularly spaced grid, useful for making data work
        with plotting using imshow.

        Parameters
        ----------
        x_list : list of numbers
            x-positions
        y_list : list of numbers
            y-positions
        z_list : list of numbers
            The property, for example distance between
            atoms, ellipticity or angle between atoms.
        """

        data_list = self._property_position(
            z_list, x_position=x_list, y_position=y_list
        )

        interpolate_x_lim = (0, self.image.shape[1])
        interpolate_y_lim = (0, self.image.shape[0])
        new_data = at._get_interpolated2d_from_unregular_data(
            data_list,
            new_x_lim=interpolate_x_lim,
            new_y_lim=interpolate_y_lim,
            upscale=upscale,
        )

        return new_data

    def get_property_map(
        self,
        x_list,
        y_list,
        z_list,
        atom_plane_list=None,
        add_zero_value_sublattice=None,
        upscale_map=2,
    ):
        """Returns an interpolated signal map of a property.

        Maps the property in z_list.
        The map is interpolated, and the upscale factor of the interpolation
        can be set (default 2).

        Parameters
        ----------
        x_list, y_list : list of numbers
            Lists of x and y positions
        z_list : list of numbers
            List of properties for positions. z[0] is the property of
            the position at x[0] and y[0]. z_list must have the same length as
            x_list and y_list.
        atom_plane_list : list of Atomap Atom_Plane objects, default None
            List of atom planes which will be added as markers in the signal.
        add_zero_value_sublattice : Sublattice
            The sublattice for A-cations in a perovskite oxide can be included
            here, when maps of oxygen tilt patterns are made. The principle is
            that another sublattice can given as a parameter, in order to add
            zero-value points in the map. This means that he atom positions in
            this sublattice will be included in the map, and the property at
            these coordinate is set to 0. This helps in the visualization of
            tilt patterns. For example, the "tilt" property" outside oxygen
            octahedron in perovskite oxides is 0. The A-cations are outside
            the octahedra.
        upscale_map : int, default 2
            Upscaling factor for the interpolated map.

        Returns
        -------
        interpolated_signal : HyperSpy Signal2D

        Examples
        --------

        Example with ellipticity as property.

        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> s = sublattice.get_property_map(
        ...                    sublattice.x_position,
        ...                    sublattice.y_position,
        ...                    sublattice.ellipticity)
        >>> s.plot()

        """
        data_scale = self.pixel_size
        if add_zero_value_sublattice is not None:
            x_list, y_list, z_list = at._add_zero_position_to_data_list(
                x_list,
                y_list,
                z_list,
                add_zero_value_sublattice.x_position,
                add_zero_value_sublattice.y_position,
            )
        data_map = self._get_regular_grid_from_unregular_property(
            x_list, y_list, z_list, upscale=upscale_map
        )
        signal = at.array2signal2d(
            data_map[2],
            self.pixel_size / upscale_map,
            units=self.units,
            rotate_flip=True,
        )
        if atom_plane_list is not None:
            marker_list = _make_atom_planes_marker_list(
                atom_plane_list, scale=data_scale, add_numbers=False
            )
            add_marker(signal, marker_list, permanent=True, plot_marker=False)
        return signal

    def _get_property_line_profile(
        self,
        x_list,
        y_list,
        z_list,
        atom_plane,
        data_scale_xy=1.0,
        data_scale_z=1.0,
        invert_line_profile=False,
        add_markers=True,
        interpolate_value=50,
    ):
        """
        Project a 2 dimensional property to a plane, and get
        values as a function of distance from this plane.
        The function will attempt to combine the data points from
        the same monolayer, to get the property values as a function
        of each monolayer. The data will be returned as an interpolation
        of these values, since HyperSpy signals currently does not support
        non-linear axes.

        The raw non-interpolated line profile data is stored in the
        output signal metadata: signal.metadata.line_profile_data,
        including the standard deviation of the values from each layer.
        The standard deviation is calculated from the variance in property
        values in each layer. So if each property in a layer has exactly
        the same value, the standard deviation will be 0.

        Parameters
        ----------
        x_list, y_list : Numpy 1D array
            x and y positions for z_list property value.
        z_list : Numpy 1D array
            The property value. x_list, y_list and z_list must have
            the same size.
        atom_plane : Atomap AtomPlane object
            The plane the data is projected onto.
        data_scale_xy : number, optional
            For scaling the x_list and y_list values.
        data_scale_z : number, optional
            For scaling the values in the z_list
        invert_line_profile : bool, optional, default False
            If True, will invert the x-axis values
        interpolate_value : int, default 50
            The amount of data points between in monolayer, due to
            HyperSpy signals not supporting non-linear axes.

        Returns
        -------
        HyperSpy signal1D

        Example
        -------
        >>> from numpy.random import random
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> for atom in sublattice.atom_list:
        ...     atom.sigma_x, atom.sigma_y = 0.5*random()+1, 0.5*random()+1
        >>> sublattice.construct_zone_axes()
        >>> x = sublattice.x_position
        >>> y = sublattice.y_position
        >>> z = sublattice.ellipticity
        >>> plane = sublattice.atom_plane_list[20]
        >>> s = sublattice._get_property_line_profile(x, y, z, plane)
        >>> s.plot()

        Accessing the raw non-interpolated data

        >>> x_list = s.metadata.line_profile_data.x_list
        >>> y_list = s.metadata.line_profile_data.y_list
        >>> std_list = s.metadata.line_profile_data.std_list

        """
        xA, yA, zA = np.array(x_list), np.array(y_list), np.array(z_list)
        xs, ys, zs = xA.shape, yA.shape, zA.shape
        if (xs != ys) or (xs != zs) or (ys != zs):
            raise ValueError("x_list, y_list and z_list must have the " "same shape")
        line_profile_data_list = self._property_position_projection(
            interface_plane=atom_plane,
            property_list=zA,
            x_position=xA,
            y_position=yA,
            scale_xy=data_scale_xy,
            scale_z=data_scale_z,
        )
        x_new = np.linspace(
            line_profile_data_list[0, 0],
            line_profile_data_list[0, -1],
            interpolate_value * len(line_profile_data_list[0]),
        )
        y_new = np.interp(
            x_new,
            line_profile_data_list[0],
            line_profile_data_list[1],
        )
        if invert_line_profile:
            x_new *= -1
        data_scale = x_new[1] - x_new[0]
        offset = x_new[0]
        signal = at.array2signal1d(y_new, scale=data_scale, offset=offset)
        if invert_line_profile:
            x_profile_list = line_profile_data_list[0] * -1
        else:
            x_profile_list = line_profile_data_list[0]
        y_profile_list = line_profile_data_list[1]
        std_profile_list = line_profile_data_list[2]
        signal.metadata.add_node("line_profile_data")
        signal.metadata.line_profile_data.x_list = x_profile_list
        signal.metadata.line_profile_data.y_list = y_profile_list
        signal.metadata.line_profile_data.std_list = std_profile_list
        if add_markers:
            marker_list = []
            for x, y in zip(x_profile_list, y_profile_list):
                marker_list.append(Point(x, y))
            add_marker(signal, marker_list, permanent=True, plot_marker=False)
        return signal

    def _get_pixel_separation(self, nearest_neighbors=2, leafsize=100):
        """
        Get the pixel separation by finding the distance between each atom
        and its two closest neighbors. From this distance list the median
        distance is found and divided by 2. This gives the distance used
        by for example atom_finding_refining.get_atom_positions

        Parameters
        ----------
        nearest_neighbor : int, optional, default 2

        Returns
        -------
        pixel_separation, int
        """
        atom_position_list = np.array([self.x_position, self.y_position]).swapaxes(0, 1)
        nearest_neighbor_data = cKDTree(atom_position_list, leafsize=leafsize)
        distance_list = []
        for atom in self.atom_list:
            nn_data_list = nearest_neighbor_data.query(
                atom.get_pixel_position(), nearest_neighbors
            )
            # Skipping the first element,
            # since it points to the atom itself
            for nn_link in nn_data_list[1][1:]:
                distance = atom.get_pixel_distance_from_another_atom(
                    self.atom_list[nn_link]
                )
                distance_list.append(distance)
        pixel_separation = np.median(distance_list) / 2
        return pixel_separation

    def find_nearest_neighbors(self, nearest_neighbors=9, leafsize=100):
        """
        Find the nearest neighbors for all the atom position objects.
        Needs to be run before doing any position refinements.

        Parameters
        ----------
        nearest_neighbors : int, default 9
        leafsize : int, default 100
        """
        atom_position_list = np.array([self.x_position, self.y_position]).swapaxes(0, 1)
        nearest_neighbor_data = sp.spatial.cKDTree(
            atom_position_list, leafsize=leafsize
        )
        for atom in self.atom_list:
            nn_data_list = nearest_neighbor_data.query(
                atom.get_pixel_position(), nearest_neighbors
            )
            nn_link_list = []
            # Skipping the first element,
            # since it points to the atom itself
            for nn_link in nn_data_list[1][1:]:
                nn_link_list.append(self.atom_list[nn_link])
            atom.nearest_neighbor_list = nn_link_list

    def get_atom_plane_slice_between_two_planes(self, atom_plane1, atom_plane2):
        """Get a list of atom planes between two atom planes.
        Both these atom planes must belong to the same zone vector.

        The list will include the two atom planes passed to the
        function.

        Parameters
        ----------
        atom_plane1, atom_plane2 : Atomap Atom_Plane object

        Returns
        -------
        atom_plane_slice : list of Atom Plane objects
        """
        atom_plane_start_index = None
        atom_plane_end_index = None
        zone_vector = atom_plane1.zone_vector
        if zone_vector != atom_plane2.zone_vector:
            raise ValueError(
                "atom_plane1 and atom_plane2 must belong to the" " same zone vector"
            )
        for index, temp_atom_plane in enumerate(
            self.atom_planes_by_zone_vector[zone_vector]
        ):
            if temp_atom_plane == atom_plane1:
                atom_plane_start_index = index
            if temp_atom_plane == atom_plane2:
                atom_plane_end_index = index + 1
        if atom_plane_start_index > atom_plane_end_index:
            temp_index = atom_plane_start_index
            atom_plane_start_index = atom_plane_end_index
            atom_plane_end_index = temp_index
        atom_plane_slice = self.atom_planes_by_zone_vector[zone_vector][
            atom_plane_start_index:atom_plane_end_index
        ]
        return atom_plane_slice

    def toggle_atom_refine_position_with_gui(self, image=None, distance_threshold=4):
        """Use GUI to toggle if atom positions will be fitted.

        Use the press atom positions with the left mouse button toggle
        if they will be fitted in the refine methods:

        - refine_atom_positions_using_2d_gaussian
        - refine_atom_positions_using_center_of_mass

        Green color means they will be fitted.
        Red color means they will not be fitted.

        Parameters
        ----------
        image : None, optional
            If None, sublattice.image will be used.
        distance_threshold : int, optional
            If a left mouse button click is within
            distance_threshold, the closest atom will be
            toggled.

        Examples
        --------
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.toggle_atom_refine_position_with_gui()

        Using a different image

        >>> image = np.random.random((300, 300))
        >>> sublattice.toggle_atom_refine_position_with_gui(image=image)

        """
        global toggle_refine_position
        if image is None:
            image = self.image
        toggle_refine_position = gui.AtomToggleRefine(
            image, self, distance_threshold=distance_threshold
        )

    def get_atom_list_between_four_atom_planes(
        self, par_atom_plane1, par_atom_plane2, ort_atom_plane1, ort_atom_plane2
    ):
        """Get a slice of atoms between two pairs of atom planes.
        Each pair must belong to the zone vector.

        Parameters
        ----------
        par_atom_plane1, par_atom_plane2 : Atomap Atom_Plane object
        ort_atom_plane1, ort_atom_plane2 : Atomap Atom_Plane object

        Returns
        -------
        atom_list : list of Atom_Position objects
        """
        ort_slice = self.get_atom_plane_slice_between_two_planes(
            ort_atom_plane1, ort_atom_plane2
        )
        par_slice = self.get_atom_plane_slice_between_two_planes(
            par_atom_plane1, par_atom_plane2
        )

        par_atom_list = []
        for atom_plane in par_slice:
            par_atom_list.extend(atom_plane.atom_list)
        ort_atom_list = []
        for temp_atom_plane in ort_slice:
            temp_atom_list = []
            for atom in temp_atom_plane.atom_list:
                if atom in par_atom_list:
                    temp_atom_list.append(atom)
            ort_atom_list.extend(temp_atom_list)
        return ort_atom_list

    def _find_perpendicular_vector(self, v):
        if v[0] == 0 and v[1] == 0:
            raise ValueError("zero vector")
        return np.cross(v, [1, 0])

    def _sort_atom_planes_by_zone_vector(self):
        for zone_vector in self.zones_axis_average_distances:
            temp_atom_plane_list = []
            for atom_plane in self.atom_plane_list:
                if atom_plane.zone_vector == zone_vector:
                    temp_atom_plane_list.append(atom_plane)
            self.atom_planes_by_zone_vector[zone_vector] = temp_atom_plane_list

        for index, (zone_vector, atom_plane_list) in enumerate(
            self.atom_planes_by_zone_vector.items()
        ):
            length = 100000000
            orthogonal_vector = (length * zone_vector[1], -length * zone_vector[0])

            closest_atom_list = []
            for atom_plane in atom_plane_list:
                closest_atom = 10000000000000000000000000
                for atom in atom_plane.atom_list:
                    dist = atom.pixel_distance_from_point(orthogonal_vector)
                    if dist < closest_atom:
                        closest_atom = dist
                closest_atom_list.append(closest_atom)
            atom_plane_list.sort(key=dict(zip(atom_plane_list, closest_atom_list)).get)

    def _remove_bad_zone_vectors(self):
        zone_vector_delete_list = []
        for zone_vector in self.atom_planes_by_zone_vector:
            atom_planes = self.atom_planes_by_zone_vector[zone_vector]
            if len(atom_planes) == 0:
                zone_vector_delete_list.append(zone_vector)
            else:
                counter_atoms = 0
                for atom_plane in atom_planes:
                    number_of_atoms = len(atom_plane.atom_list)
                    if number_of_atoms == 2:
                        counter_atoms += 1
                ratio = counter_atoms / len(atom_planes)
                if ratio > 0.6:
                    atom_planes_delete_list = []
                    for atom_plane in atom_planes:
                        for atom in atom_plane.atom_list:
                            atom.in_atomic_plane.remove(atom_plane)
                        atom_planes_delete_list.append(atom_plane)
                    zone_vector_delete_list.append(zone_vector)
                    for atom_plane in atom_planes_delete_list:
                        self.atom_plane_list.remove(atom_plane)
        for zone_vector in zone_vector_delete_list:
            del self.atom_planes_by_zone_vector[zone_vector]
            for i, v in enumerate(self.zones_axis_average_distances):
                if v == zone_vector:
                    self.zones_axis_average_distances_names.pop(i)
            self.zones_axis_average_distances.remove(zone_vector)

    def refine_atom_positions_using_2d_gaussian(
        self,
        image_data=None,
        percent_to_nn=None,
        mask_radius=None,
        rotation_enabled=True,
        show_progressbar=True,
    ):
        """
        Use 2D-Gaussian fitting to refine the atom positions on the image
        data.

        Parameters
        ----------
        image_data : 2D NumPy array, optional
            If this is not specified, original_image will be used.
        percent_to_nn : float, default 0.4
            Percent to nearest neighbor. The function will find the closest
            nearest neighbor to the current atom position, and
            this value times percent_to_nn will be the radius of the mask
            centered on the atom position. Value should be somewhere
            between 0.01 (1%) and 1 (100%).
        mask_radius : float, optional
            Radius of the mask around each atom. If this is not set,
            the radius will be the distance to the nearest atom in the
            same sublattice times the `percent_to_nn` value.
            Note: if `mask_radius` is not specified, the Atom_Position objects
            must have a populated nearest_neighbor_list.
        rotation_enabled : bool, default True
            If True, rotation will be enabled for the 2D Gaussians.
            This will most likely make the fitting better, but potentially
            at a cost of less robust fitting.
        show_progressbar : default True

        Example
        -------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.find_nearest_neighbors()
        >>> sublattice.refine_atom_positions_using_2d_gaussian()

        See Also
        --------
        refine_atom_positions_using_center_of_mass

        """
        if (mask_radius is not None) and (percent_to_nn is not None):
            raise ValueError(
                "Both percent_to_nn and mask_radius is specified, "
                "only one of them should be set"
            )
        if mask_radius is None:
            if percent_to_nn is None:
                percent_to_nn = 0.4
            self._check_if_nearest_neighbor_list()
        if image_data is None:
            image_data = self.original_image
        image_data = image_data.astype("float64")
        for atom in progressbar(
            self.atom_list, desc="Gaussian fitting", disable=not show_progressbar
        ):
            if atom.refine_position:
                atom.refine_position_using_2d_gaussian(
                    image_data,
                    rotation_enabled=rotation_enabled,
                    percent_to_nn=percent_to_nn,
                    mask_radius=mask_radius,
                )

    def refine_atom_positions_using_center_of_mass(
        self,
        image_data=None,
        percent_to_nn=None,
        mask_radius=None,
        show_progressbar=True,
    ):
        """
        Use center of mass to refine the atom positions on the image
        data.

        Parameters
        ----------
        image_data : 2D NumPy array, optional
            If this is not specified, original_image will be used.
        percent_to_nn : float, default 0.25
            Percent to nearest neighbor. The function will find the closest
            nearest neighbor to the current atom position, and
            this value times percent_to_nn will be the radius of the mask
            centered on the atom position. Value should be somewhere
            between 0.01 (1%) and 1 (100%).
        mask_radius : float, optional
        show_progressbar : default True

        Example
        -------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.find_nearest_neighbors()
        >>> sublattice.refine_atom_positions_using_center_of_mass()

        See Also
        --------
        refine_atom_positions_using_2d_gaussian

        """
        if (mask_radius is not None) and (percent_to_nn is not None):
            raise ValueError(
                "Both percent_to_nn and mask_radius is specified, "
                "only one of them should be set"
            )
        if mask_radius is None:
            if percent_to_nn is None:
                percent_to_nn = 0.25
            self._check_if_nearest_neighbor_list()
        if image_data is None:
            image_data = self.original_image
        image_data = image_data.astype("float64")
        for atom in progressbar(
            self.atom_list, desc="Center of mass", disable=not show_progressbar
        ):
            if atom.refine_position:
                atom.refine_position_using_center_of_mass(
                    image_data, percent_to_nn=percent_to_nn, mask_radius=mask_radius
                )

    def get_nearest_neighbor_directions(self, pixel_scale=True, neighbors=None):
        """
        Get the vector to the nearest neighbors for the atoms
        in the sublattice. Giving information similar to a FFT
        of the image, but for real space.

        Useful for seeing if the peakfinding and symmetry
        finder worked correctly. Potentially useful for
        doing structure fingerprinting.

        Parameters
        ----------
        pixel_scale : bool, optional. Default True
            If True, will return coordinates in pixel scale.
            If False, will return in data scale (pixel_size).
        neighbors : int, optional
            The number of neighbors returned for each atoms.
            If no number is given, will return all the neighbors,
            which is typically 9 for each atom. As given when
            running the symmetry finder.

        Returns
        -------
        Position : tuple
            (x_position, y_position). Where both are NumPy arrays.

        Examples
        --------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.construct_zone_axes()
        >>> x_pos, y_pos = sublattice.get_nearest_neighbor_directions()
        >>> import matplotlib.pyplot as plt
        >>> cax = plt.scatter(x_pos, y_pos)

        With all the keywords

        >>> x_pos, y_pos = sublattice.get_nearest_neighbor_directions(
        ...     pixel_scale=False, neighbors=3)
        """
        if neighbors is None:
            neighbors = 10000

        x_pos_distances = []
        y_pos_distances = []
        for atom in self.atom_list:
            for index, neighbor_atom in enumerate(atom.nearest_neighbor_list):
                if index > neighbors:
                    break
                distance = atom.get_pixel_difference(neighbor_atom)
                if not ((distance[0] == 0) and (distance[1] == 0)):
                    x_pos_distances.append(distance[0])
                    y_pos_distances.append(distance[1])
        if not pixel_scale:
            scale = self.pixel_size
        else:
            scale = 1.0
        x_pos_distances = np.array(x_pos_distances) * scale
        y_pos_distances = np.array(y_pos_distances) * scale
        return (x_pos_distances, y_pos_distances)

    def get_nearest_neighbor_directions_all(self):
        """
        Like get_nearest_neighbour_directions(), but considers
        all other atoms (instead of the typical 9) as neighbors
        from each atom.

        This method also does not require atoms to have the
        atom.nearest_neighbor_list parameter populated with
        sublattice.find_nearest_neighbors().

        Without the constraint of looking at only n nearest neighbours,
        blazing fast internal NumPy functions can be utilized to
        calculate directions. However, memory usage will grow quadratically
        with the number of atomic columns. E.g.:
        1000 atomic columns will require ~8MB of memory.
        10,000 atomic columns will require ~800MB of memory.
        100,000 atomic columns will throw a MemoryError exception
        on most machines.

        Returns
        -------
        nn_position_array : NumPy array
            In the form [x_pos, y_pos].

        Examples
        --------
        >>> import numpy as np
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> x_pos, y_pos = sublattice.get_nearest_neighbor_directions_all()
        >>> mask = np.sqrt(x_pos**2 + y_pos**2) < 3
        >>> import matplotlib.pyplot as plt
        >>> cax = plt.scatter(x_pos[mask], y_pos[mask])
        """

        n_atoms = len(self.atom_list)

        # Calculate the offset matrix
        #
        # Note: The terms 'direction', 'offset' and 'distance vector' are used
        # interchangeably in this method.
        x_array = np.asarray(self.x_position, dtype=np.float32)
        y_array = np.asarray(self.y_position, dtype=np.float32)
        dx = x_array - x_array[..., np.newaxis]
        dy = y_array - y_array[..., np.newaxis]

        # Assert statements here are just to help the reader understand what's
        # going on by keeping track of the shapes of arrays used.
        assert dx.shape == (n_atoms, n_atoms)
        assert dy.shape == (n_atoms, n_atoms)

        # Produce a mask that selects all elements except the diagonal
        # i.e. distance vectors from an atom to itself.
        mask = ~np.diag([True] * n_atoms)
        assert mask.shape == (n_atoms, n_atoms)

        # Remove the diagonal and flatten
        nn = np.array([dx[mask], dy[mask]])
        assert nn.shape == (2, n_atoms * (n_atoms - 1))

        return nn

    def _make_translation_symmetry(self, pixel_separation_factor=7):
        """Find the major translation symmetries for the atom positions.

        Essentially finds the (x, y) vectors between all the atom positions,
        and uses nearest neighbor statistics to generate a list of the
        nearest ones. I.e. the most high-index zone axes for the positions.

        Only the unique vectors are kept, meaning parallel and antiparallel
        ones are removed.

        This function populates sublattice.zones_axis_average_distances and
        sublattice.zones_axis_average_distances_names

        Note: this function requires sublattice._pixel_separation to be set
        before running this function. This is normally done by calling
        sublattice._get_pixel_separation()

        Parameters
        ----------
        pixel_separation_factor : default 7

        Example
        -------
        >>> import atomap.dummy_data as dd
        >>> sublattice = dd.get_simple_cubic_sublattice()
        >>> sublattice._pixel_separation = sublattice._get_pixel_separation()
        >>> sublattice._make_translation_symmetry()
        >>> zone_vectors = sublattice.zones_axis_average_distances

        See also
        --------
        get_fingerprint_2d : function used to get full vector list

        """
        pixel_radius = self._pixel_separation * pixel_separation_factor
        fp_2d = self.get_fingerprint_2d(pixel_radius=pixel_radius)
        clusters = []
        for zone_vector in fp_2d:
            cluster = (
                float(format(zone_vector[0], ".2f")),
                float(format(zone_vector[1], ".2f")),
            )
            clusters.append(cluster)
        clusters = _remove_parallel_vectors(
            clusters, distance_tolerance=self._pixel_separation / 1.5
        )

        new_zone_vector_name_list = []
        for zone_vector in clusters:
            new_zone_vector_name_list.append(str(tuple(zone_vector)))

        self.zones_axis_average_distances = clusters
        self.zones_axis_average_distances_names = new_zone_vector_name_list

    def _get_atom_plane_list_from_zone_vector(self, zone_vector):
        temp_atom_plane_list = []
        for atom_plane in self.atom_plane_list:
            if atom_plane.zone_vector == zone_vector:
                temp_atom_plane_list.append(atom_plane)
        return temp_atom_plane_list

    def _generate_all_atom_plane_list(self, atom_plane_tolerance=0.5):
        for zone_vector in self.zones_axis_average_distances:
            self._find_all_atomic_planes_from_direction(
                zone_vector, atom_plane_tolerance=atom_plane_tolerance
            )

    def _find_all_atomic_planes_from_direction(
        self, zone_vector, atom_plane_tolerance=0.5
    ):
        for atom in self.atom_list:
            if not atom.is_in_atomic_plane(zone_vector):
                atom_plane = self._find_atomic_columns_from_atom(
                    atom, zone_vector, atom_plane_tolerance=atom_plane_tolerance
                )
                if not (len(atom_plane) == 1):
                    atom_plane_instance = Atom_Plane(atom_plane, zone_vector, self)
                    for atom in atom_plane:
                        atom.in_atomic_plane.append(atom_plane_instance)
                    self.atom_plane_list.append(atom_plane_instance)

    def _find_atomic_columns_from_atom(
        self, start_atom, zone_vector, atom_plane_tolerance=0.5
    ):
        atom_range = atom_plane_tolerance * self._pixel_separation
        end_of_atom_plane = False
        zone_axis_list1 = [start_atom]
        while not end_of_atom_plane:
            atom = zone_axis_list1[-1]
            atoms_within_distance = []
            for neighbor_atom in atom.nearest_neighbor_list:
                distance = neighbor_atom.pixel_distance_from_point(
                    point=(atom.pixel_x + zone_vector[0], atom.pixel_y + zone_vector[1])
                )
                if distance < atom_range:
                    atoms_within_distance.append([distance, neighbor_atom])
            if atoms_within_distance:
                atoms_within_distance.sort()
                zone_axis_list1.append(atoms_within_distance[0][1])
            if zone_axis_list1[-1] is atom:
                end_of_atom_plane = True
                atom._end_atom.append(zone_vector)

        zone_vector2 = (-1 * zone_vector[0], -1 * zone_vector[1])
        start_of_atom_plane = False
        zone_axis_list2 = [start_atom]
        while not start_of_atom_plane:
            atom = zone_axis_list2[-1]
            atoms_within_distance = []
            for neighbor_atom in atom.nearest_neighbor_list:
                distance = neighbor_atom.pixel_distance_from_point(
                    point=(
                        atom.pixel_x + zone_vector2[0],
                        atom.pixel_y + zone_vector2[1],
                    )
                )
                if distance < atom_range:
                    atoms_within_distance.append([distance, neighbor_atom])
            if atoms_within_distance:
                atoms_within_distance.sort()
                zone_axis_list2.append(atoms_within_distance[0][1])
            if zone_axis_list2[-1] is atom:
                start_of_atom_plane = True
                atom._start_atom.append(zone_vector)

        if not (len(zone_axis_list2) == 1):
            zone_axis_list1.extend(zone_axis_list2[1:])
        return zone_axis_list1

    def find_missing_atoms_from_zone_vector(
        self,
        zone_vector,
        vector_fraction=0.5,
        extend_outer_edges=False,
        outer_edge_limit=5,
    ):
        """Returns a list of coordinates between atoms given by a zone vector.

        These coordinates are given by a point between adjacent atoms
        in the atom planes with the given zone_vector.

        Parameters
        ----------
        zone_vector : tuple
            Zone vector for the atom planes where the new atoms are positioned
            between the atoms in the sublattice.
        vector_fraction : float, optional
            Fraction of the distance between the adjacent atoms.
            Value between 0 and 1, default 0.5
        extend_outer_edge : bool
            If True, atoms at the edges of the sublattice will also
            be included. Default False.
        outer_edge_limit : 5
            Will only matter if extend_outer_edge is True. If the edge atoms
            are too close to the edge of the image data, they will
            not be included in the output. In pixel values, default 5.
            Higher value means fewer atoms are included.

        Returns
        -------
        xy_coordinates : List of tuples

        Example
        -------
        >>> import atomap.api as am
        >>> sublattice_A = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice_A.construct_zone_axes()
        >>> zone_axis = sublattice_A.zones_axis_average_distances[0]
        >>> B_pos = sublattice_A.find_missing_atoms_from_zone_vector(
        ...                       zone_axis)

        Using the vector_fraction parameter

        >>> s = am.dummy_data.get_hexagonal_double_signal()
        >>> peaksA = am.get_atom_positions(s, separation=10)
        >>> sublatticeA = am.Sublattice(peaksA, s.data)
        >>> sublatticeA.construct_zone_axes()
        >>> zv = sublatticeA.zones_axis_average_distances[5]
        >>> peaksB = sublatticeA.find_missing_atoms_from_zone_vector(
        ...     zv, vector_fraction=0.7)
        >>> sublatticeB = am.Sublattice(peaksB, s.data)
        >>> sublatticeB.plot()

        Use extend_outer_edges to also get the second sublattice atoms
        which are not between the first sublattice's atoms

        >>> peaksB = sublatticeA.find_missing_atoms_from_zone_vector(
        ...     zv, vector_fraction=0.7, extend_outer_edges=True)
        >>> sublatticeB = am.Sublattice(peaksB, s.data)

        """
        atom_plane_list = self.atom_planes_by_zone_vector[zone_vector]
        if extend_outer_edges:
            im_y, im_x = self.image.shape
            im_x -= outer_edge_limit
            im_y -= outer_edge_limit
        new_atom_list = []
        for atom_plane in atom_plane_list:
            for atom_index, atom in enumerate(atom_plane.atom_list[1:]):
                previous_atom = atom_plane.atom_list[atom_index]
                difference_vector = previous_atom.get_pixel_difference(atom)
                new_atom_x = (
                    previous_atom.pixel_x - difference_vector[0] * vector_fraction
                )
                new_atom_y = (
                    previous_atom.pixel_y - difference_vector[1] * vector_fraction
                )
                new_atom_list.append((new_atom_x, new_atom_y))
                if extend_outer_edges:
                    if previous_atom is atom_plane.start_atom:
                        new_atom_x = previous_atom.pixel_x - difference_vector[0] * (
                            vector_fraction - 1
                        )
                        new_atom_y = previous_atom.pixel_y - difference_vector[1] * (
                            vector_fraction - 1
                        )
                        if (
                            outer_edge_limit < new_atom_x < im_x
                            and outer_edge_limit < new_atom_y < im_y
                        ):
                            new_atom_list.append((new_atom_x, new_atom_y))
                    if atom is atom_plane.end_atom:
                        new_atom_x = (
                            atom.pixel_x - difference_vector[0] * vector_fraction
                        )
                        new_atom_y = (
                            atom.pixel_y - difference_vector[1] * vector_fraction
                        )
                        if (
                            outer_edge_limit < new_atom_x < im_x
                            and outer_edge_limit < new_atom_y < im_y
                        ):
                            new_atom_list.append((new_atom_x, new_atom_y))
        return new_atom_list

    def get_atom_planes_on_image(
        self, atom_plane_list, image=None, add_numbers=True, color="red"
    ):
        """
        Get atom_planes signal as lines on the image.

        Parameters
        ----------
        atom_plane_list : list of atom_plane objects
            atom_planes to be plotted on the image contained in
        image : 2D Array, optional
        add_numbers : bool, optional, default True
            If True, will the number of the atom plane at the end of the
            atom plane line. Useful for finding the index of the atom plane.
        color : string, optional, default red
            The color of the lines and text used to show the atom planes.

        Returns
        -------
        HyperSpy signal2D object

        Examples
        --------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.construct_zone_axes()
        >>> zone_vector = sublattice.zones_axis_average_distances[0]
        >>> atom_planes = sublattice.atom_planes_by_zone_vector[zone_vector]
        >>> s = sublattice.get_atom_planes_on_image(atom_planes)
        >>> s.plot()
        """
        if image is None:
            image = self.original_image
        marker_list = _make_atom_planes_marker_list(
            atom_plane_list, add_numbers=add_numbers, scale=self.pixel_size, color=color
        )
        signal = at.array2signal2d(image, self.pixel_size, self.units)
        add_marker(signal, marker_list, permanent=True, plot_marker=False)
        return signal

    def get_all_atom_planes_by_zone_vector(
        self, zone_vector_list=None, image=None, add_numbers=True, color="red"
    ):
        """
        Get a overview of atomic planes for some or all zone vectors.

        Parameters
        ----------
        zone_vector_list : optional
            List of zone vectors for visualizing atomic planes.
            Default is visualizing all the zone vectors.
        image : 2D Array, optional
        add_numbers : bool, optional, default True
            If True, will the number of the atom plane at the end of the
            atom plane line. Useful for finding the index of the atom plane.
        color : string, optional, default red
            The color of the lines and text used to show the atom planes.

        Returns
        -------
        HyperSpy signal2D object if given a single zone vector,
        list of HyperSpy signal2D if given a list (or none) of zone vectors.

        Examples
        --------

        Getting a list signals showing the atomic planes for all the
        zone vectors

        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.construct_zone_axes()
        >>> s = sublattice.get_all_atom_planes_by_zone_vector()
        >>> s.plot()

        Several zone vectors

        >>> zone_vec_list = sublattice.zones_axis_average_distances[0:3]
        >>> s = sublattice.get_all_atom_planes_by_zone_vector(zone_vec_list)
        >>> s.plot()

        Different image

        >>> from numpy.random import random
        >>> im = random((9, 9))
        >>> s = sublattice.get_all_atom_planes_by_zone_vector(image=im)
        >>> s.plot()
        """
        if image is None:
            image = self.original_image
        if zone_vector_list is None:
            self._check_if_zone_axis_list()
            zone_vector_list = self.zones_axis_average_distances

        atom_plane_list = []
        for zone_vector in zone_vector_list:
            atom_plane_list.append(self.atom_planes_by_zone_vector[zone_vector])
        marker_list = _make_multidim_atom_plane_marker_list(
            atom_plane_list, scale=self.pixel_size
        )
        signal = at.array2signal2d(image, self.pixel_size, self.units)
        signal = hs.stack([signal] * len(zone_vector_list), show_progressbar=False)
        add_marker(signal, marker_list, permanent=True, plot_marker=False)
        signal.metadata.General.title = "Atom planes by zone vector"
        signal_ax0 = signal.axes_manager.signal_axes[0]
        signal_ax1 = signal.axes_manager.signal_axes[1]
        x = signal_ax0.index2value(int(image.shape[0] * 0.1))
        y = signal_ax1.index2value(int(image.shape[1] * 0.1))
        text_marker_list = _make_zone_vector_text_marker_list(
            zone_vector_list, x=x, y=y
        )
        add_marker(signal, text_marker_list, permanent=True, plot_marker=False)
        return signal

    def get_atom_list_on_image(
        self, atom_list=None, image=None, color=None, add_numbers=False, markersize=20
    ):
        """
        Plot atom positions on the image data.

        Parameters
        ----------
        atom_list : list of Atom objects, optional
            Atom positions to plot. If no list is given,
            will use the atom_list.
        image : 2-D NumPy array, optional
            Image data for plotting. If none is given, will use
            the original_image.
        color : string, optional
        add_numbers : bool, default False
            Plot the number of the atom beside each atomic
            position in the plot. Useful for locating
            misfitted atoms.
        markersize : number, default 20
            Size of the atom position markers

        Returns
        -------
        HyperSpy 2D-signal
            The atom positions as permanent markers stored in the metadata.

        Examples
        --------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.construct_zone_axes()
        >>> s = sublattice.get_atom_list_on_image()
        >>> s.plot()

        Number all the atoms

        >>> s = sublattice.get_atom_list_on_image(add_numbers=True)
        >>> s.plot()

        Plot a subset of the atom positions

        >>> atoms = sublattice.atom_list[0:20]
        >>> s = sublattice.get_atom_list_on_image(
        ...     atom_list=atoms, add_numbers=True)
        >>> s.plot(cmap='viridis')

        Saving the signal as HyperSpy HDF5 file, which saves the atom
        positions as permanent markers.

        >>> s = sublattice.get_atom_list_on_image()
        >>> s.save("sublattice_atom_positions.hdf5", overwrite=True)
        """
        if color is None:
            color = self._plot_color
        if image is None:
            image = self.original_image
        if atom_list is None:
            atom_list = self.atom_list
        marker_list = _make_atom_position_marker_list(
            atom_list,
            scale=self.pixel_size,
            color=color,
            markersize=markersize,
            add_numbers=add_numbers,
        )
        signal = at.array2signal2d(image, self.pixel_size, self.units)
        add_marker(signal, marker_list, permanent=True, plot_marker=False)

        return signal

    def get_ellipticity_vector(
        self, image=None, atom_plane_list=None, vector_scale=1.0, color="red"
    ):
        """
        Visualize the ellipticity and direction of the atomic columns
        using markers in a HyperSpy signal.

        Parameters
        ----------
        image : 2-D NumPy array, optional
        atom_plane_list : list of AtomPlane instances
        vector_scale : scaling of the vector
        color : string

        Returns
        -------
        HyperSpy 2D-signal with the ellipticity vectors as
        permanent markers

        Examples
        --------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> for atom in sublattice.atom_list:
        ...     atom.sigma_x, atom.sigma_y = 1., 1.2
        >>> s = sublattice.get_ellipticity_vector(vector_scale=20)
        >>> s.plot()
        """
        if image is None:
            image = self.original_image
        elli_list = []
        for atom in self.atom_list:
            elli_rot = atom.get_ellipticity_vector()
            elli_list.append(
                [
                    atom.pixel_x,
                    atom.pixel_y,
                    elli_rot[0] * vector_scale,
                    elli_rot[1] * vector_scale,
                ]
            )
        signal = at.array2signal2d(image, self.pixel_size, self.units)
        marker_list = _make_arrow_marker_list(
            elli_list, scale=self.pixel_size, color=color
        )
        if atom_plane_list is not None:
            marker_list.extend(
                _make_atom_planes_marker_list(
                    atom_plane_list, scale=self.pixel_size, add_numbers=False
                )
            )
        add_marker(signal, marker_list, permanent=True, plot_marker=False)
        return signal

    def integrate_column_intensity(
        self,
        method="Voronoi",
        max_radius="Auto",
        data_to_integrate=None,
        remove_edge_cells=False,
        edge_pixels=1,
    ):
        """Integrate signal around the atoms in the sublattice

        If the sublattice is a part of an Atom_Lattice object, this function
        will only take into account the atoms contained in this sublattice.
        To get the integrated intensity for all the sublattices, use the
        integrate_column_intensity in the Atom_Lattice object.

        See atomap.tools.integrate for more information about the parameters.

        Parameters
        ----------
        method : string
            Voronoi or Watershed
        max_radius : int, optional
        data_to_integrate : NumPy array, HyperSpy signal or array-like
            Works with 2D, 3D and 4D arrays, so for example an EEL spectrum
            image can be used.
        remove_edge_cells : bool
            Determine whether to replace the cells touching the signal edge
            with np.nan values, which makes automatic contrast estimation
            easier later.
        edge_pixels : int
            Only used if remove_edge_cells is True. Determines the number of
            pixels from the border to remove.

        Returns
        -------
        i_points, i_record, p_record

        Examples
        --------
        >>> import atomap.api as am
        >>> sl = am.dummy_data.get_simple_cubic_sublattice(image_noise=True)
        >>> i_points, i_record, p_record = sl.integrate_column_intensity()

        See also
        --------
        tools.integrate

        """
        if data_to_integrate is None:
            data_to_integrate = self.image
        i_points, i_record, p_record = at.integrate(
            data_to_integrate,
            self.x_position,
            self.y_position,
            method=method,
            max_radius=max_radius,
            remove_edge_cells=remove_edge_cells,
            edge_pixels=edge_pixels,
        )
        return (i_points, i_record, p_record)

    def get_atom_column_amplitude_max_intensity(self, image=None, percent_to_nn=0.40):
        """Finds the maximal intensity for each atomic column.

        Finds the maximal image intensity of each atomic column inside
        an area covering the atomic column.

        Results are stored in each Atom_Position object as
        amplitude_max_intensity, which can most easily be accessed in
        through the sublattice object (see the examples below).

        Parameters
        ----------
        image : NumPy 2D array, default None
            Uses original_image by default.
        percent_to_nn : float, default 0.4
            Determines the boundary of the area surrounding each atomic
            column, as fraction of the distance to the nearest neighbour.

        Example
        -------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.find_nearest_neighbors()
        >>> sublattice.get_atom_column_amplitude_max_intensity()
        >>> intensity_list = sublattice.atom_amplitude_max_intensity

        """
        self._check_if_nearest_neighbor_list()
        if image is None:
            image = self.original_image
        percent_distance = percent_to_nn
        for atom in self.atom_list:
            atom.calculate_max_intensity(image, percent_to_nn=percent_distance)

    def get_atom_column_amplitude_min_intensity(self, image=None, percent_to_nn=0.40):
        """Finds the minimum intensity for each atomic column.

        Finds the minimum image intensity of each atomic column inside
        an area covering the atomic column.

        Results are stored in each Atom_Position object as
        amplitude_min_intensity, which can most easily be accessed in
        through the sublattice object (see the examples below).

        Parameters
        ----------
        image : NumPy 2D array, default None
            Uses original_image by default.
        percent_to_nn : float, default 0.4
            Determines the boundary of the area surrounding each atomic
            column, as fraction of the distance to the nearest neighbour.

        Example
        -------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.find_nearest_neighbors()
        >>> sublattice.get_atom_column_amplitude_min_intensity()
        >>> intensity_list = sublattice.atom_amplitude_min_intensity

        """
        self._check_if_nearest_neighbor_list()
        if image is None:
            image = self.original_image

        percent_distance = percent_to_nn
        for atom in self.atom_list:
            atom.calculate_min_intensity(image, percent_to_nn=percent_distance)

    def get_atom_list_atom_amplitude_gauss2d_range(self, amplitude_range):
        atom_list = []
        for atom in self.atom_list:
            if atom.amplitude_gaussian > amplitude_range[0]:
                if atom.amplitude_gaussian < amplitude_range[1]:
                    atom_list.append(atom)
        return atom_list

    def save_map_from_datalist(
        self,
        data_list,
        data_scale,
        atom_plane=None,
        dtype="float32",
        signal_name="datalist_map.hdf5",
    ):
        """data_list : numpy array, 4D"""
        im = hs.signals.Signal2D(data_list[2])
        x_scale = data_list[0][1][0] - data_list[0][0][0]
        y_scale = data_list[1][0][1] - data_list[1][0][0]
        im.axes_manager[0].scale = x_scale * data_scale
        im.axes_manager[1].scale = y_scale * data_scale
        im.change_dtype("float32")
        if not (atom_plane is None):
            im.metadata.add_node("marker.atom_plane.x")
            im.metadata.add_node("marker.atom_plane.y")
            im.metadata.marker.atom_plane.x = atom_plane.get_x_position_list()
            im.metadata.marker.atom_plane.y = atom_plane.get_y_position_list()
        im.save(signal_name, overwrite=True)

    def _get_distance_and_position_list_between_atom_planes(
        self, atom_plane0, atom_plane1
    ):
        list_x, list_y, list_z = [], [], []
        for atom in atom_plane0.atom_list:
            pos_x, pos_y = atom_plane1.get_closest_position_to_point(
                (atom.pixel_x, atom.pixel_y), extend_line=True
            )
            distance = atom.pixel_distance_from_point(point=(pos_x, pos_y))
            list_x.append((pos_x + atom.pixel_x) * 0.5)
            list_y.append((pos_y + atom.pixel_y) * 0.5)
            list_z.append(distance)
        data_list = np.array([list_x, list_y, list_z])
        return data_list

    def get_ellipticity_line_profile(
        self, atom_plane, invert_line_profile=False, interpolate_value=50
    ):
        """
        Returns a line profile of the atoms ellipticity.

        This gives the ellipticity as a function of the distance from a
        given atom plane (interface).

        The raw data can be accessed in
        s.metadata.line_profile_data, both the position (x_list),
        ellipticity (y_list), and the standard deviation of the ellipticity
        for each layer (std_list).

        Parameters
        ----------
        atom_plane : Atomap AtomPlane object
            The plane which is defined as the 0-point in the spatial
            dimension.
        invert_line_profile : bool, optional, default False
            Passed to _get_property_line_profile(). If True, will invert the
            x-axis values.
        interpolate_value : int, default 50
            Passed to _get_property_line_profile(). The amount of data points
            between in monolayer, due to HyperSpy signals not supporting
            non-linear axes.

        Returns
        -------
        signal : HyperSpy Signal1D

        Example
        -------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.construct_zone_axes()
        >>> zone = sublattice.zones_axis_average_distances[1]
        >>> plane = sublattice.atom_planes_by_zone_vector[zone][4]
        >>> s_elli_line = sublattice.get_ellipticity_line_profile(plane)

        Getting the raw line profile data

        >>> position = s_elli_line.metadata.line_profile_data.x_list
        >>> ellipticity = s_elli_line.metadata.line_profile_data.y_list
        >>> std = s_elli_line.metadata.line_profile_data.std_list

        """
        signal = self._get_property_line_profile(
            self.x_position,
            self.y_position,
            self.ellipticity,
            atom_plane,
            data_scale_xy=self.pixel_size,
            invert_line_profile=invert_line_profile,
            interpolate_value=interpolate_value,
        )
        return signal

    def get_ellipticity_map(self, upscale_map=2.0, atom_plane_list=None):
        """
        Get a HyperSpy signal showing the magnitude of the ellipticity
        for the sublattice.

        Parameters
        ----------
        upscale_map : number, default 2.0
            Amount of upscaling compared to the original image given
            to Atomap. Note, a high value here can greatly increase
            the memory use for large images.
        atom_plane_list : List of Atomap AtomPlane object, optional
            If a list of AtomPlanes are given, the plane positions
            will be added to the signal as permanent markers. Which
            can be visualized using s.plot(plot_markers=True).
            Useful for showing the location of for example an interface.

        Returns
        -------
        HyperSpy 2D signal

        Examples
        --------
        >>> from numpy.random import random
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> for atom in sublattice.atom_list:
        ...     atom.sigma_x, atom.sigma_y = 0.5*random()+1, 0.5*random()+1
        >>> s_elli = sublattice.get_ellipticity_map()
        >>> s_elli.plot()

        Include an atom plane, which is added to the signal as a marker

        >>> sublattice.construct_zone_axes()
        >>> atom_plane = [sublattice.atom_plane_list[10]]
        >>> s_elli = sublattice.get_ellipticity_map(atom_plane_list=atom_plane)
        >>> s_elli.plot()
        """
        signal = self.get_property_map(
            self.x_position,
            self.y_position,
            self.ellipticity,
            atom_plane_list=atom_plane_list,
            upscale_map=upscale_map,
        )
        title = "Sublattice {} ellipticity".format(self.name)
        signal.metadata.General.title = title
        return signal

    def get_monolayer_distance_line_profile(
        self, zone_vector, atom_plane, invert_line_profile=False, interpolate_value=50
    ):
        """
        Finds the distance between each atom and the next monolayer, and the
        distance to atom_plane (the 0-point). The monolayers belong to the
        zone vector zone_vector. For more information on the definition of
        monolayer distance, check
        sublattice.get_monolayer_distance_list_from_zone_vector()

        The raw data can be accessed in
        s.metadata.line_profile_data, both the position (x_list),
        distance (y_list), and the standard deviation of the distance
        for each layer (std_list).

        Parameters
        ----------

        zone_vector : tuple
            Zone vector for the monolayers for which the separation will be
            found
        atom_plane : Atomap AtomPlane object
            Passed to get_monolayer_distance_list_from_zone_vector().
            0-point in the line profile.
        invert_line_profile : bool, optional, default False
            Passed to _get_property_line_profile(). If True, will invert the
            x-axis values.
        interpolate_value : int, default 50
            Passed to _get_property_line_profile(). The amount of data points
            between in monolayer, due to HyperSpy signals not supporting
            non-linear axes.

        Returns
        -------
        HyperSpy signal1D

        Example
        -------
        >>> from numpy.random import random
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> for atom in sublattice.atom_list:
        ...     atom.sigma_x, atom.sigma_y = 0.5*random()+1, 0.5*random()+1
        >>> sublattice.construct_zone_axes()
        >>> zone = sublattice.zones_axis_average_distances[0]
        >>> plane = sublattice.atom_planes_by_zone_vector[zone][0]
        >>> s = sublattice.get_monolayer_distance_line_profile(zone, plane)

        Getting the raw line profile data

        >>> position = s.metadata.line_profile_data.x_list
        >>> distance = s.metadata.line_profile_data.y_list
        >>> std = s.metadata.line_profile_data.std_list

        """
        self._check_if_zone_axis_list()
        data_list = self.get_monolayer_distance_list_from_zone_vector(zone_vector)
        signal = self._get_property_line_profile(
            data_list[0],
            data_list[1],
            data_list[2],
            atom_plane,
            data_scale_xy=self.pixel_size,
            data_scale_z=self.pixel_size,
            invert_line_profile=invert_line_profile,
            interpolate_value=interpolate_value,
        )
        return signal

    def get_monolayer_distance_map(
        self, zone_vector_list=None, atom_plane_list=None, upscale_map=2
    ):
        self._check_if_zone_axis_list()
        zone_vector_index_list = self._get_zone_vector_index_list(zone_vector_list)
        zone_vector_list = []
        signal_list = []
        for zone_index, zone_vector in zone_vector_index_list:
            signal_title = "Monolayer distance {}".format(zone_index)
            data_list = self.get_monolayer_distance_list_from_zone_vector(zone_vector)
            signal = self.get_property_map(
                data_list[0], data_list[1], data_list[2], upscale_map=upscale_map
            )
            signal.metadata.General.Title = signal_title
            signal_list.append(signal)
            zone_vector_list.append(zone_vector)

        if len(signal_list) == 1:
            signal = signal_list[0]
        else:
            signal = hs.stack(signal_list, show_progressbar=False)
        if atom_plane_list is not None:
            marker_list = _make_atom_planes_marker_list(
                atom_plane_list, scale=self.pixel_size, add_numbers=False
            )
            add_marker(signal, marker_list, permanent=True, plot_marker=False)
        signal_ax0 = signal.axes_manager.signal_axes[0]
        signal_ax1 = signal.axes_manager.signal_axes[1]
        x = signal_ax0.index2value(int(signal_ax0.high_index * 0.1))
        y = signal_ax1.index2value(int(signal_ax1.high_index * 0.1))
        text_marker_list = _make_zone_vector_text_marker_list(
            zone_vector_list, x=x, y=y
        )
        add_marker(signal, text_marker_list, permanent=True, plot_marker=False)
        title = "Sublattice {} monolayer distance".format(self.name)
        signal.metadata.General.title = title
        return signal

    def get_atom_distance_map(
        self,
        zone_vector_list=None,
        atom_plane_list=None,
        prune_outer_values=False,
        invert_line_profile=False,
        add_zero_value_sublattice=None,
        upscale_map=2,
    ):
        self._check_if_zone_axis_list()
        zone_vector_index_list = self._get_zone_vector_index_list(zone_vector_list)

        signal_list = []
        zone_vector_list = []
        for zone_index, zone_vector in zone_vector_index_list:
            signal_title = "Atom distance {}".format(zone_index)
            data_list = self.get_atom_distance_list_from_zone_vector(zone_vector)

            signal = self.get_property_map(
                data_list[0],
                data_list[1],
                data_list[2],
                add_zero_value_sublattice=add_zero_value_sublattice,
                upscale_map=upscale_map,
            )
            signal.metadata.General.Title = signal_title
            signal_list.append(signal)
            zone_vector_list.append(zone_vector)

        if len(signal_list) == 1:
            signal = signal_list[0]
        else:
            signal = hs.stack(signal_list, show_progressbar=False)
        if atom_plane_list is not None:
            marker_list = _make_atom_planes_marker_list(
                atom_plane_list, scale=self.pixel_size, add_numbers=False
            )
            add_marker(signal, marker_list, permanent=True, plot_marker=False)
        signal_ax0 = signal.axes_manager.signal_axes[0]
        signal_ax1 = signal.axes_manager.signal_axes[1]
        x = signal_ax0.index2value(int(signal_ax0.high_index * 0.1))
        y = signal_ax1.index2value(int(signal_ax1.high_index * 0.1))
        text_marker_list = _make_zone_vector_text_marker_list(
            zone_vector_list, x=x, y=y
        )
        add_marker(signal, text_marker_list, permanent=True, plot_marker=False)
        title = "Sublattice {} atom distance".format(self.name)
        signal.metadata.General.title = title
        return signal

    def get_atom_distance_difference_line_profile(
        self, zone_vector, atom_plane, invert_line_profile=False, interpolate_value=50
    ):
        self._check_if_zone_axis_list()
        data_list = self.get_atom_distance_difference_from_zone_vector(zone_vector)
        signal = self._get_property_line_profile(
            data_list[0],
            data_list[1],
            data_list[2],
            atom_plane,
            data_scale_xy=self.pixel_size,
            data_scale_z=self.pixel_size,
            invert_line_profile=invert_line_profile,
            interpolate_value=interpolate_value,
        )
        return signal

    def get_atom_distance_difference_map(
        self,
        zone_vector_list=None,
        atom_plane_list=None,
        prune_outer_values=False,
        invert_line_profile=False,
        add_zero_value_sublattice=None,
        upscale_map=2,
    ):
        self._check_if_zone_axis_list()
        zone_vector_index_list = self._get_zone_vector_index_list(zone_vector_list)
        zone_vector_list = []
        signal_list = []
        for zone_index, zone_vector in zone_vector_index_list:
            data_list = self.get_atom_distance_difference_from_zone_vector(zone_vector)
            if len(data_list[2]) != 0:
                signal = self.get_property_map(
                    data_list[0],
                    data_list[1],
                    data_list[2],
                    add_zero_value_sublattice=add_zero_value_sublattice,
                    upscale_map=upscale_map,
                )
                signal_list.append(signal)
                zone_vector_list.append(zone_vector)

        if len(signal_list) == 1:
            signal = signal_list[0]
        else:
            signal = hs.stack(signal_list, show_progressbar=False)
        if atom_plane_list is not None:
            marker_list = _make_atom_planes_marker_list(
                atom_plane_list, scale=self.pixel_size, add_numbers=False
            )
            add_marker(signal, marker_list, permanent=True, plot_marker=False)
        signal_ax0 = signal.axes_manager.signal_axes[0]
        signal_ax1 = signal.axes_manager.signal_axes[1]
        x = signal_ax0.index2value(int(signal_ax0.high_index * 0.1))
        y = signal_ax1.index2value(int(signal_ax1.high_index * 0.1))
        text_marker_list = _make_zone_vector_text_marker_list(
            zone_vector_list, x=x, y=y
        )
        add_marker(signal, text_marker_list, permanent=True, plot_marker=False)
        title = "Sublattice {} atom distance difference".format(self.name)
        signal.metadata.General.title = title
        return signal

    def get_model_image(
        self, image_shape=None, sigma_quantile=5, show_progressbar=True
    ):
        """
        Generate an image of the atomic positions from the
        atom positions Gaussian parameter's.

        Parameters
        ----------
        image_shape : tuple (x, y), optional
            Size of the image generated. Note that atoms might be
            outside the image if (x, y) is too small.
        show_progressbar : default True

        Returns
        -------
        signal, HyperSpy 2D signal

        """
        if image_shape is None:
            model_image = np.zeros(self.image.shape)
        else:
            model_image = np.zeros(image_shape[::-1])
        X, Y = np.meshgrid(
            np.arange(model_image.shape[1]), np.arange(model_image.shape[0])
        )

        g = Gaussian2D(
            centre_x=0.0, centre_y=0.0, sigma_x=1.0, sigma_y=1.0, rotation=1.0, A=1.0
        )

        im_y, im_x = model_image.shape
        for atom in progressbar(self.atom_list, disable=not show_progressbar):
            x, y = atom.pixel_x, atom.pixel_y
            sx, sy = atom.sigma_x, atom.sigma_y
            atom_slice = atom._get_atom_slice(im_x, im_y, sigma_quantile=sigma_quantile)
            Xa, Ya = X[atom_slice], Y[atom_slice]
            g.A.value = atom.amplitude_gaussian
            g.centre_x.value = x
            g.centre_y.value = y
            g.sigma_x.value = sx
            g.sigma_y.value = sy
            g.rotation.value = atom.rotation
            model_image[atom_slice] += g.function(Xa, Ya)
        s = Signal2D(model_image)
        return s

    def _get_zone_vector_index_list(self, zone_vector_list):
        if zone_vector_list is None:
            zone_vector_list = self.zones_axis_average_distances

        zone_vector_index_list = []
        for zone_vector in zone_vector_list:
            for index, temp_zone_vector in enumerate(self.zones_axis_average_distances):
                if temp_zone_vector == zone_vector:
                    zone_index = index
                    break
            zone_vector_index_list.append([zone_index, zone_vector])
        return zone_vector_index_list

    def _plot_debug_start_end_atoms(self):
        for zone_index, zone_vector in enumerate(self.zones_axis_average_distances):
            fig, ax = plt.subplots(figsize=(10, 10))
            cax = ax.imshow(self.image)
            if self._plot_clim:
                cax.set_clim(self._plot_clim[0], self._plot_clim[1])
            for atom_index, atom in enumerate(self.atom_list):
                if zone_vector in atom._start_atom:
                    ax.plot(atom.pixel_x, atom.pixel_y, "o", color="blue")
                    ax.text(atom.pixel_x, atom.pixel_y, str(atom_index))
            for atom_index, atom in enumerate(self.atom_list):
                if zone_vector in atom._end_atom:
                    ax.plot(atom.pixel_x, atom.pixel_y, "o", color="green")
                    ax.text(atom.pixel_x, atom.pixel_y, str(atom_index))
            ax.set_ylim(0, self.image.shape[0])
            ax.set_xlim(0, self.image.shape[1])
            fig.tight_layout()
            fig.savefig("debug_plot_start_end_atoms_zone" + str(zone_index) + ".jpg")

    def _plot_atom_position_convergence(self, figname="atom_position_convergence.jpg"):
        position_absolute_convergence = []
        position_jump_convergence = []
        for atom in self.atom_list:
            dist0 = atom.get_position_convergence(distance_to_first_position=True)
            dist1 = atom.get_position_convergence()
            position_absolute_convergence.append(dist0)
            position_jump_convergence.append(dist1)

        absolute_convergence = np.array(position_absolute_convergence).mean(axis=0)
        relative_convergence = np.array(position_jump_convergence).mean(axis=0)

        fig, axarr = plt.subplots(2, 1, sharex=True)
        absolute_ax = axarr[0]
        relative_ax = axarr[1]

        absolute_ax.plot(absolute_convergence)
        relative_ax.plot(relative_convergence)

        absolute_ax.set_ylabel("Average distance from start")
        relative_ax.set_ylabel("Average jump pr. iteration")
        relative_ax.set_xlabel("Refinement step")

        fig.tight_layout()
        fig.savefig(self.name + "_" + figname)

    def get_zone_vector_mean_angle(self, zone_vector):
        """Get the mean angle between the atoms planes with a
        specific zone vector and the horizontal axis. For each
        atom plane the angle between all the atoms, its
        neighbor and horizontal axis is calculated.
        The mean of these angles for all the atom
        planes is returned.
        """
        atom_plane_list = self.atom_planes_by_zone_vector[zone_vector]
        angle_list = []
        for atom_plane in atom_plane_list:
            temp_angle_list = atom_plane.get_angle_to_horizontal_axis()
            angle_list.extend(temp_angle_list)
        mean_angle = np.array(angle_list).mean()
        return mean_angle

    def construct_zone_axes(self, atom_plane_tolerance=0.5, zone_axis_para_list=False):
        """Constructs the zone axes for an atomic resolution image.

        The zone axes are found by finding the 15 nearest neighbors for each
        atom position in the sublattice, and finding major translation
        symmetries among the nearest neighbours. Only unique zone axes are
        kept, and "bad" ones are removed.

        After finding the zone axes, atom planes are constructed.

        Parameters
        ----------
        atom_plane_tolerance : scalar, default 0.5
            When constructing the atomic planes, the method will try to locate
            the atoms by "jumping" one zone vector, and seeing if there is an
            atom within the pixel_separation times atom_plane_tolerance. So
            this value should be increased if the atomic planes are
            non-continuous and "split".
        zone_axis_para_list : parameter list or bool, default False
            A zone axes parameter list is used to name and index the zone
            axes. See atomap.process_parameters for more info. Useful for
            automation.

        Example
        -------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice
        <Sublattice,  (atoms:400,planes:0)>
        >>> sublattice.construct_zone_axes()
        >>> sublattice
        <Sublattice,  (atoms:400,planes:4)>

        See also
        --------
        atom_finding_refining.construct_zone_axes_from_sublattice

        """
        construct_zone_axes_from_sublattice(
            self,
            atom_plane_tolerance=atom_plane_tolerance,
            zone_axis_para_list=zone_axis_para_list,
        )

    def _get_fingerprint(self, pixel_radius=100):
        """
        Produce a Fingerprint class object.

        Example
        -------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> fp = sublattice._get_fingerprint()
        >>> fp_distance = fp.fingerprint_
        >>> fp_vector = fp.cluster_centers_
        """

        n_atoms = len(self.atom_list)

        # Get distance vectors to all neighbouring atoms from each atom
        x_pos, y_pos = self.get_nearest_neighbor_directions_all()

        # Assert statements here are just to help the reader understand the
        # algorithm by keeping track of the shapes of arrays used.
        assert x_pos.shape == y_pos.shape == (n_atoms * (n_atoms - 1),)

        # Produce a mask that only select vectors that are shorter than radius
        mask = (x_pos ** 2 + y_pos ** 2) < pixel_radius ** 2
        assert mask.shape == (n_atoms * (n_atoms - 1),)
        n_atoms_closer_than_radius = mask.sum()

        # Apply mask to get nearest neighbours
        nn = np.array([x_pos[mask], y_pos[mask]])
        assert nn.shape == (
            2,
            n_atoms_closer_than_radius,
        )

        # Apply the fingerprinter
        fingerprinter = at.Fingerprinter()
        fingerprinter.fit(nn.T)

        return fingerprinter

    def get_fingerprint_2d(self, pixel_radius=100):
        """
        Produce a distance and direction fingerprint of the sublattice.

        Parameters
        ----------
        pixel_radius : int, default 100

        Returns
        -------
        cluster_array : NumPy array

        Example
        -------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> fp = sublattice.get_fingerprint_2d()
        >>> import matplotlib.pyplot as plt
        >>> cax = plt.scatter(fp[:,0], fp[:,1], marker='o')

        """
        fingerprinter = self._get_fingerprint(pixel_radius=pixel_radius)
        return fingerprinter.cluster_centers_

    def get_fingerprint_1d(self, pixel_radius=100):
        """
        Produce a distance fingerprint of the sublattice.

        Example
        -------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> fp = sublattice.get_fingerprint_1d()
        >>> import matplotlib.pyplot as plt
        >>> cax = plt.plot(fp, marker='o')
        """
        fingerprinter = self._get_fingerprint(pixel_radius=pixel_radius)
        return fingerprinter.fingerprint_

    def get_position_history(
        self,
        image=None,
        color="red",
        add_numbers=False,
        markersize=20,
        show_progressbar=True,
    ):
        """
        Plot position history of each atom positions on the image data.

        Parameters
        ----------
        image : 2-D NumPy array, optional
            Image data for plotting. If none is given, will use
            the original_image.
        color : string, default 'red'
        add_numbers : bool, default False
            Plot the number of the atom beside each atomic
            position in the plot. Useful for locating
            misfitted atoms.
        markersize : number, default 20
            Size of the atom position markers
        show_progressbar : default True

        Returns
        -------
        HyperSpy 2D-signal
            The atom positions as permanent markers stored in the metadata.

        """
        if image is None:
            image = self.original_image

        pos_num = len(self.atom_list[0].old_pixel_x_list) + 1
        if pos_num == 1:
            s = self.get_atom_list_on_image(
                image=image, color=color, add_numbers=add_numbers, markersize=markersize
            )
            return s

        atom_num = len(self.atom_list)
        peak_list = np.zeros((pos_num, atom_num, 2))
        for i, atom in enumerate(self.atom_list):
            for j in range(len(atom.old_pixel_x_list)):
                peak_list[j, i, 0] = atom.old_pixel_x_list[j]
                peak_list[j, i, 1] = atom.old_pixel_y_list[j]
                peak_list[-1, i, 0] = atom.pixel_x
                peak_list[-1, i, 1] = atom.pixel_y

        signal = Signal2D(image)
        s = hs.stack([signal] * pos_num, show_progressbar=False)

        marker_list_x = np.ones((len(peak_list), atom_num)) * -100
        marker_list_y = np.ones((len(peak_list), atom_num)) * -100

        for index, peaks in enumerate(peak_list):
            marker_list_x[index, 0 : len(peaks)] = peaks[:, 0]
            marker_list_y[index, 0 : len(peaks)] = peaks[:, 1]

        marker_list = []
        for i in progressbar(
            range(marker_list_x.shape[1]), disable=not show_progressbar
        ):
            m = hs.markers.point(
                x=marker_list_x[:, i], y=marker_list_y[:, i], color="red"
            )
            marker_list.append(m)

        s.add_marker(marker_list, permanent=True, plot_marker=False)
        return s

    def _get_sublattice_atom_list_mask(self, image_data=None, radius=1):
        """
        Returns a list of indices, where mask_list[0] is the indices
        for a circular mask with the given radius around the atom in
        atom_list[0].

        Parameters
        ----------
        image_data : NumPy 2D array
            Image data from which the shape of the mask array is found.
            By default, the sublattice.original_image is used.
        radius : float
            radius of the circular mask

        Returns
        -------
        A list of mask indices lists. Mask_list has the same length as
        atom_list, and each element in the list is an array of indices

        """
        if image_data is None:
            image_data = self.original_image
        mask_list = []
        for atom in self.atom_list:
            centerX, centerY = atom.pixel_x, atom.pixel_y
            temp_mask = _make_circular_mask(
                centerY, centerX, image_data.shape[0], image_data.shape[1], radius
            )
            indices = np.where(temp_mask)
            mask_list.append(indices)
        return mask_list

    def mask_image_around_sublattice(self, image_data=None, radius=2):
        """
        Returns a HyperSpy signal containing a masked image. The mask covers
        the area around each atom position in the sublattice, from a given
        radius away from the center position of the atom. This radius is given
        in pixels.

        Parameters
        ----------
        image_data : 2-D NumPy array, optional
            Image data for plotting. If none is given, will use
            the original_image.

        radius : int, optional
            The radius in pixels away from the atom centre pixels, determining
            the area that shall not be masked. The default radius is 2 pixels.

        Returns
        -------
        masked_signal : HyperSpy 2D-signal

        Examples
        --------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> s = sublattice.mask_image_around_sublattice(radius=3)
        >>> s.plot()

        """
        if image_data is None:
            image_data = self.original_image
        mask = np.full_like(image_data, False)
        for atom in self.atom_list:
            centerX, centerY = atom.pixel_x, atom.pixel_y
            temp_mask = _make_circular_mask(
                centerY, centerX, image_data.shape[0], image_data.shape[1], radius
            )
            mask[np.where(temp_mask)] = True
        s = hs.signals.Signal2D(mask * image_data)
        return s

    def find_sublattice_intensity_from_masked_image(self, image_data=None, radius=2):
        """Find the image intensity of each atomic column in the sublattice.

        The intensity of the atomic column is given by the average intensity
        of the pixels inside an area within a radius in pixels from each
        atom position.

        Parameters
        ----------
        image_data : 2-D NumPy array, optional
            Image data for plotting. If none is given, will use
            the original_image.

        radius : int, optional, default 2
            The radius in pixels away from the atom centre pixels, determining
            the area that shall not be masked. The default radius is 3 pixels.

        Examples
        --------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.find_sublattice_intensity_from_masked_image(
        ...     sublattice.image, 3)
        >>> intensity = sublattice.intensity_mask

        """
        if image_data is None:
            image_data = self.original_image
        for atom in self.atom_list:
            atom.find_atom_intensity_inside_mask(image_data, radius)

    def plot(self, color=None, add_numbers=False, markersize=20, **kwargs):
        """
        Plot all atom positions in the sublattice on the image data.

        The sublattice.original_image attribute is used as the image.

        Parameters
        ----------
        color : string, optional
            Color of the atom positions. If none is specific the value
            set in sublattice._plot_color is used.
        add_numbers : bool, default False
            Plot the number of the atom beside each atomic position in the
            plot. Useful for locating misfitted atoms.
        markersize : number, default 20
            Size of the atom position markers
        **kwargs
            Addition keywords passed to HyperSpy's signal plot function.

        Examples
        --------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.plot()

        Setting color and color map

        >>> sublattice.plot(color='green', cmap='viridis', add_numbers=True,
        ...                 markersize=15)

        See also
        --------
        get_atom_list_on_image : get HyperSpy signal with atom positions
                                 as markers. More customizability.

        """
        signal = self.get_atom_list_on_image(
            color=color, add_numbers=add_numbers, markersize=markersize
        )
        signal.plot(**kwargs, plot_markers=True)

    def plot_planes(self, image=None, add_numbers=True, color="red", **kwargs):
        """
        Show the atomic planes for all zone vectors.

        Parameters
        ----------
        image : 2D Array, optional
            If image None, the original image of the sublattice is used as
            background image.
        add_numbers : bool, optional, default True
            If True, will the number of the atom plane at the end of the
            atom plane line. Useful for finding the index of the atom plane.
        color : string, optional, default red
            The color of the lines and text used to show the atom planes.
        **kwargs
            Additional keywords passed to HyperSpy's signal plot function.

        Examples
        --------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.construct_zone_axes()
        >>> sublattice.plot_planes()
        """

        signal = self.get_all_atom_planes_by_zone_vector(
            zone_vector_list=None, image=image, add_numbers=add_numbers, color=color
        )
        signal.plot(**kwargs)

    def plot_ellipticity_vectors(self, save=False):
        """Get a quiver plot of rotation and ellipticity for the sublattice.

        sublattice.refine_atom_positions_using_2d_gaussian has to be run
        at least once before using this function.
        If the sublattice hasn't been refined with 2D-Gaussians, the value
        for ellipticity and rotation are default, 1 and 0 respectively.
        When sigma_x and sigma_y are equal (circle) the ellipticity is 1.
        To better visualize changes in ellipticity, the 0-point for
        ellipticity in the plot is set to circular atomic columns.

        Parameters
        ----------
        save : bool
            If true, the figure is saved as 'vector_field.png'.

        Examples
        --------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_fantasite_sublattice()
        >>> sublattice.construct_zone_axes()
        >>> sublattice.refine_atom_positions_using_2d_gaussian()
        >>> sublattice.plot_ellipticity_vectors()

        """
        ellipticity = np.asarray(self.ellipticity) - 1
        rot = -np.asarray(self.rotation_ellipticity)
        u_quiver = ellipticity * np.cos(rot)
        v_quiver = ellipticity * np.sin(rot)
        u_quiver *= -1

        plot_vector_field(
            self.x_position, self.y_position, u_quiver, v_quiver, save=save
        )

    def plot_ellipticity_map(self, **kwargs):
        """
        Plot the magnitude of the ellipticity.

        Parameters
        ----------
        **kwargs
            Addition keywords passed to HyperSpy's signal plot function.

        Examples
        --------
        >>> import atomap.api as am
        >>> sublattice = am.dummy_data.get_fantasite_sublattice()
        >>> sublattice.construct_zone_axes()
        >>> sublattice.refine_atom_positions_using_2d_gaussian()
        >>> sublattice.plot_ellipticity_map()

        """
        s_elli = self.get_ellipticity_map()
        s_elli.plot(**kwargs)

    def get_middle_position_list(self, zone_axis0, zone_axis1):
        """Find the middle point between all four neighboring atoms.

        The neighbors are found by moving one step along the atom planes
        belonging to zone_axis0 and zone_axis1.

        So atom planes must be constructed first.

        Parameters
        ----------
        sublattice : Sublattice object
        za0 : tuple
        za1 : tuple

        Return
        ------
        middle_position_list : list

        Examples
        --------
        >>> import atomap.analysis_tools as an
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.construct_zone_axes()
        >>> za0 = sublattice.zones_axis_average_distances[0]
        >>> za1 = sublattice.zones_axis_average_distances[1]
        >>> middle_position_list = an.get_middle_position_list(
        ...     sublattice, za0, za1)

        """
        middle_position_list = an.get_middle_position_list(self, zone_axis0, zone_axis1)
        return middle_position_list

    def get_polarization_from_second_sublattice(
        self, zone_axis0, zone_axis1, sublattice, color="cyan"
    ):
        """Get a signal showing the polarization using a second sublattice.

        Parameters
        ----------
        zone_axis0 : tuple
        zone_axis1 : tuple
        sublattice : Sublattice object
        color : string, optional
            Default 'cyan'.

        Returns
        -------
        signal_polarization : HyperSpy Signal2D
            The vector data is stored in s.metadata.vector_list. These
            are visualized using the plot() method.

        Examples
        --------
        >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
        >>> sublatticeA = atom_lattice.sublattice_list[0]
        >>> sublatticeB = atom_lattice.sublattice_list[1]
        >>> sublatticeA.construct_zone_axes()
        >>> za0, za1 = sublatticeA.zones_axis_average_distances[0:2]
        >>> s_p = sublatticeA.get_polarization_from_second_sublattice(
        ...     za0, za1, sublatticeB, color='blue')
        >>> s_p.plot()
        >>> vector_list = s_p.metadata.vector_list

        """
        middle_pos_list = self.get_middle_position_list(zone_axis0, zone_axis1)
        vector_list = an.get_vector_shift_list(sublattice, middle_pos_list)
        marker_list = vector_list_to_marker_list(
            vector_list, color=color, scale=self.pixel_size
        )
        signal = at.array2signal2d(self.image, self.pixel_size, self.units)
        signal.add_marker(marker_list, permanent=True, plot_marker=False)
        signal.metadata.add_node("vector_list")
        signal.metadata.vector_list = vector_list
        return signal

    def estimate_local_scanning_distortion(self, image=None, radius=6, edge_skip=2):
        """Get the amount of local scanning distortion from atomic columns.

        This is done by assuming the atomic columns has a symmetrical shape,
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
        s_distortion_x, s_distortion_y, distortion_x_mean, distortion_y_mean
            Both horizontal and vertical directions. For standard raster scans,
            horizontal will be the fast scan direction,
            and y the slow scan direction. Typically the slow scan direction
            has the largest amount of distortion.

        Examples
        --------

        >>> sl = am.dummy_data.get_scanning_distortion_sublattice()
        >>> s_x, s_y, avg_x, avg_y = sl.estimate_local_scanning_distortion()
        >>> s_x.plot()
        >>> s_y.plot()

        """
        if image is None:
            image = self.image
        x_list, y_list, dx_list, dy_list = [], [], [], []
        for atom in self.atom_list:
            distortions = atom.estimate_local_scanning_distortion(
                image_data=image, radius=radius, edge_skip=edge_skip
            )
            x_list.append(atom.pixel_x)
            y_list.append(atom.pixel_y)
            dx_list.append(distortions[0])
            dy_list.append(distortions[1])

        s_distortion_x = self.get_property_map(x_list, y_list, dx_list)
        s_distortion_y = self.get_property_map(x_list, y_list, dy_list)
        dist_x_mean = np.mean(dx_list)
        dist_y_mean = np.mean(dy_list)
        return s_distortion_x, s_distortion_y, dist_x_mean, dist_y_mean

    def pair_distribution_function(self, image=None, n_bins=200, rel_range=0.5):
        """
        Plots a two dimensional pair distribution function (PDF) from the atoms
        contained in the sublattice.

        The intensity of peaks in the PDF is corrected to account for missing
        information (i.e. the fact atoms are present outside of the field of
        view) and differences in area at different distances.

        Parameters
        ----------
        image : 2D HyperSpy Signal object
        n_bins : int
            Number of bins to use for the PDF.
        rel_range : float
            The range of the PDF as a fraction of the field of view of the
            image.

        Returns
        -------
        s_pdf : HyperSpy Signal 1D Object
            The calculated PDF.

        Examples
        --------
        >>> s = am.dummy_data.get_simple_cubic_signal()
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> s_pdf = sublattice.pair_distribution_function(s)
        >>> s_pdf.plot()

        """
        if image is None:
            image = self.signal

        pdf = an.pair_distribution_function(
            image, self.atom_positions, n_bins, rel_range
        )
        return pdf

    def set_scale(self, scale, units):
        """Set the scale for the sublattice.

        In units per pixel.

        Parameters
        ----------
        scale : float
            Preferably Ångstrøm. If the distance between the atom columns
            are 4 Ångstrøm, and there are 20 pixels between them. The scale should be
            0.2.
        units : string

        Example
        -------
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.set_scale(0.2, "Å")

        """
        self.pixel_size = scale
        self.units = units

    def set_element_info(self, element, z):
        """Set which atoms are present along atomic columns for all atoms in
        the sublattice. This will set all atomic columns to have the same
        atoms present. If you want to set elements for each atom individually
        see Atom_Position.set_element_info().

        Parameters
        ----------
        element : str or list of str
            elements contained in the atomic column.
        z : list of floats
            Must be in Ångstrøm

        Examples
        --------
        >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice.set_element_info("C", [0, 2.5])
        >>> sublattice2 = am.dummy_data.get_simple_cubic_sublattice()
        >>> sublattice2.set_element_info(["C", "O"], [0, 2.5])
        """

        [atom.set_element_info(element, z) for atom in self.atom_list]
