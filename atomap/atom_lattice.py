from hyperspy.external.progressbar import progressbar
import numpy as np
from hyperspy.signals import Signal2D
import atomap.atom_finding_refining as afr
import atomap.plotting as pl
import atomap.tools as at
from ase import Atoms


class Atom_Lattice:
    def __init__(
        self,
        image=None,
        name="",
        sublattice_list=None,
        original_image=None,
        image_extra=None,
    ):
        """
        Parameters
        ----------
        image : 2D NumPy array, optional
        sublattice_list : list of sublattice object, optional
        original_image : 2D NumPy array, optional
        image_extra : 2D NumPy array, optional

        Attributes
        ----------
        image : 2D NumPy array
        original_image : 2D NumPy array
        x_position : list of floats
            x positions for all sublattices.
        y_position : list of floats
            y positions for all sublattices.
        image_extra : 2D NumPy array, optional

        """
        if not isinstance(name, str):
            raise ValueError("name must be string, not {0}".format(type([])))
        if sublattice_list is None:
            self.sublattice_list = []
        else:
            self.sublattice_list = sublattice_list
        at._image_init(self, image=image, original_image=original_image)
        self.image_extra = image_extra
        self.name = name
        self._pixel_separation = 10
        self._original_filename = ""

    def __repr__(self):
        return "<%s, %s (sublattice(s): %s)>" % (
            self.__class__.__name__,
            self.name,
            len(self.sublattice_list),
        )

    @property
    def x_position(self):
        x_list = []
        for subl in self.sublattice_list:
            x_list.append(subl.x_position)
        x_pos = np.concatenate(x_list)
        return x_pos

    @property
    def y_position(self):
        y_list = []
        for subl in self.sublattice_list:
            y_list.append(subl.y_position)
        y_pos = np.concatenate(y_list)
        return y_pos

    @property
    def signal(self):
        if self.image is None:
            assert ValueError("image has not been set")
        s = Signal2D(self.image)
        if self.sublattice_list:
            sublattice = self.get_sublattice(0)
            s.axes_manager.signal_axes[0].scale = sublattice.pixel_size
            s.axes_manager.signal_axes[1].scale = sublattice.pixel_size
        return s

    def get_sublattice(self, sublattice_id):
        """
        Get a sublattice object from either sublattice index
        or name.
        """
        if isinstance(sublattice_id, str):
            for sublattice in self.sublattice_list:
                if sublattice.name == sublattice_id:
                    return sublattice
        elif isinstance(sublattice_id, int):
            return self.sublattice_list[sublattice_id]
        raise ValueError("Could not find sublattice " + str(sublattice_id))

    def _construct_zone_axes_for_sublattices(self, sublattice_list=None):
        if sublattice_list is None:
            sublattice_list = self.sublattice_list
        for sublattice in sublattice_list:
            afr.construct_zone_axes_from_sublattice(sublattice)

    def integrate_column_intensity(
        self,
        method="Voronoi",
        max_radius="Auto",
        data_to_integrate=None,
        remove_edge_cells=False,
        edge_pixels=1,
    ):
        """Integrate signal around the atoms in the atom lattice.

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
        >>> al = am.dummy_data.get_simple_atom_lattice_two_sublattices()
        >>> i_points, i_record, p_record = al.integrate_column_intensity()

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

    def get_sublattice_atom_list_on_image(
        self, image=None, add_numbers=False, markersize=20
    ):
        if image is None:
            if self.original_image is None:
                image = self.image
            else:
                image = self.original_image
        marker_list = []
        scale = self.sublattice_list[0].pixel_size
        for sublattice in self.sublattice_list:
            marker_list.extend(
                pl._make_atom_position_marker_list(
                    sublattice.atom_list,
                    scale=scale,
                    color=sublattice._plot_color,
                    markersize=markersize,
                    add_numbers=add_numbers,
                )
            )
        signal = at.array2signal2d(image, scale, self.units)
        signal.add_marker(marker_list, permanent=True, plot_marker=False)

        return signal

    def set_scale(self, scale, units):
        """Set the scale for the Atom_Lattice and the Sublattices.

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
        >>> atom_lattice = am.dummy_data.get_simple_atom_lattice_two_sublattices()
        >>> atom_lattice.set_scale(0.2, "Å")

        """
        self.pixel_size = scale
        self.units = units
        for sublattice in self.sublattice_list:
            sublattice.pixel_size = scale
            sublattice.units = units

    def convert_to_ase(self):
        """
        Convert the Atomlattice object to an Atoms object of the Atomic
        Simulation Environment package. All Sublattices must have element_info
        set. All sublattices must have the correct scale set (in Angstroms), and
        the z positions set with sublattice.set_element_info must be in Angstrom.

        Returns
        -------
        atoms : ASE Atoms object

        Examples
        --------
        >>> atom_lattice = am.dummy_data.get_perovskite_001_atom_lattice()
        >>> atoms = atom_lattice.convert_to_ase()

        """
        if not hasattr(self.sublattice_list[0].atom_list[0], "element_info"):
            raise AttributeError(
                "Atom position is missing element_info, each sublattice need "
                "to have run 'set_element_info'. For example: "
                'atom_lattice.sublattice_list[0].set_element_info("C", [0.])'
            )
        atoms = Atoms()
        for sublattice in self.sublattice_list:
            for atom_column in sublattice.atom_list:
                for atom in atom_column.element_info:
                    new_atom = Atoms(
                        atom_column.element_info[atom],
                        positions=[
                            (
                                atom_column.pixel_x * sublattice.pixel_size,
                                atom_column.pixel_y * sublattice.pixel_size,
                                atom,
                            )
                        ],
                    )
                    atoms += new_atom
        return atoms

    def save(self, filename=None, overwrite=False):
        """
        Save the AtomLattice object as an HDF5 file.
        This will store the information about the individual atomic
        positions, like the pixel position, sigma and rotation.
        The input image(s) and modified version of these will also
        be saved.

        Parameters
        ----------
        filename : string, optional
        overwrite : bool, default False

        Examples
        --------
        >>> from numpy.random import random
        >>> sl = am.dummy_data.get_simple_cubic_sublattice()
        >>> atom_lattice = am.Atom_Lattice(random((9, 9)), "test", [sl])
        >>> atom_lattice.save("test.hdf5", overwrite=True)

        Loading the atom lattice:

        >>> atom_lattice1 = am.load_atom_lattice_from_hdf5("test.hdf5")

        """
        from atomap.io import save_atom_lattice_to_hdf5

        if filename is None:
            filename = self.name + "_atom_lattice.hdf5"
        save_atom_lattice_to_hdf5(self, filename=filename, overwrite=overwrite)

    def plot(self, image=None, add_numbers=False, markersize=20, **kwargs):
        """
        Plot all atom positions for all sub lattices on the image data.

        The Atom_Lattice.original_image is used as the image. For the
        sublattices, sublattice._plot_color is used as marker color.
        This color is set when the sublattice is initialized, but it can
        also be changed.

        Parameters
        ----------
        image : 2D NumPy array, optional
        add_numbers : bool, default False
            Plot the number of the atom beside each atomic position in the
            plot. Useful for locating misfitted atoms.
        markersize : number, default 20
            Size of the atom position markers
        **kwargs
            Addition keywords passed to HyperSpy's signal plot function.

        Examples
        --------
        >>> import atomap.testing_tools as tt
        >>> test_data = tt.MakeTestData(50, 50)
        >>> import numpy as np
        >>> test_data.add_atom_list(np.arange(5, 45, 5), np.arange(5, 45, 5))
        >>> atom_lattice = test_data.atom_lattice
        >>> atom_lattice.plot()

        Change sublattice colour and color map

        >>> atom_lattice.sublattice_list[0]._plot_color = 'green'
        >>> atom_lattice.plot(cmap='viridis')

        See also
        --------
        get_sublattice_atom_list_on_image : get HyperSpy signal with atom
            positions as markers. More customizability.
        """
        signal = self.get_sublattice_atom_list_on_image(
            image=image, add_numbers=add_numbers, markersize=markersize
        )
        signal.plot(**kwargs, plot_markers=True)


class Dumbbell_Lattice(Atom_Lattice):
    def __init__(self, sublattice_list=None, *args, **kwargs):
        if sublattice_list is not None:
            if len(sublattice_list) != 2:
                raise ValueError(
                    "sublattice_list must contain two sublattices,"
                    " not {0}".format(len(sublattice_list))
                )
            n_atoms0 = len(sublattice_list[0].atom_list)
            n_atoms1 = len(sublattice_list[1].atom_list)
            if n_atoms0 != n_atoms1:
                raise ValueError("Both sublattices must have the same number of atoms")

        super().__init__(sublattice_list=sublattice_list, *args, **kwargs)

    @property
    def dumbbell_x(self):
        sub0 = self.sublattice_list[0]
        sub1 = self.sublattice_list[1]
        x_array = np.mean((sub0.x_position, sub1.x_position), axis=0)
        return x_array

    @property
    def dumbbell_y(self):
        sub0 = self.sublattice_list[0]
        sub1 = self.sublattice_list[1]
        y_array = np.mean((sub0.y_position, sub1.y_position), axis=0)
        return y_array

    @property
    def dumbbell_distance(self):
        sub0 = self.sublattice_list[0]
        sub1 = self.sublattice_list[1]
        distance_list = []
        for i_atom in range(len(sub0.atom_list)):
            atom0 = sub0.atom_list[i_atom]
            atom1 = sub1.atom_list[i_atom]
            distance = atom0.get_pixel_distance_from_another_atom(atom1)
            distance_list.append(distance)
        return np.array(distance_list)

    @property
    def dumbbell_angle(self):
        sub0 = self.sublattice_list[0]
        sub1 = self.sublattice_list[1]
        angle_list = []
        for i_atom in range(len(sub0.atom_list)):
            atom0 = sub0.atom_list[i_atom]
            atom1 = sub1.atom_list[i_atom]
            angle = atom0.get_angle_between_atoms(atom1)
            angle_list.append(angle)
        return np.array(angle_list)

    def get_dumbbell_intensity_difference(self, radius=4, image=None):
        """Get the difference of intensity between the atoms in the dumbbells.

        The intensity of the atom is calculated by getting a the mean intensity
        of a disk around the position of each atom, given by the radius
        parameter.

        Parameters
        ----------
        radius : int
            Default 4
        image : array-like, optional

        Returns
        -------
        intensity_difference_list : NumPy array

        Examples
        --------
        >>> dl = am.dummy_data.get_dumbbell_heterostructure_dumbbell_lattice()
        >>> intensity_difference = dl.get_dumbbell_intensity_difference()

        """
        if image is None:
            if self.original_image is None:
                image = self.image
            else:
                image = self.original_image
        sub0 = self.sublattice_list[0]
        sub1 = self.sublattice_list[1]
        intensity_difference_list = []
        for i_atom in range(len(sub0.atom_list)):
            atom0 = sub0.atom_list[i_atom]
            atom1 = sub1.atom_list[i_atom]
            atom0.find_atom_intensity_inside_mask(image, radius)
            atom1.find_atom_intensity_inside_mask(image, radius)
            intensity_difference = atom0.intensity_mask - atom1.intensity_mask
            intensity_difference_list.append(intensity_difference)
        return np.array(intensity_difference_list)

    def refine_position_gaussian(
        self, image=None, show_progressbar=True, percent_to_nn=0.40, mask_radius=None
    ):
        """Fit several atoms at the same time.

        For datasets where the atoms are too close together to do the fitting
        individually.

        Parameters
        ----------
        image : NumPy 2D array, optional
        show_progressbar : bool, default True
        percent_to_nn : scalar
            Default 0.4
        mask_radius : float, optional
            Radius of the mask around each atom. If this is not set,
            the radius will be the distance to the nearest atom in the
            same sublattice times the `percent_to_nn` value.
            Note: if `mask_radius` is not specified, the Atom_Position objects
            must have a populated nearest_neighbor_list. This is normally done
            through the sublattice class, but can also be done manually.

        Examples
        --------
        >>> dl = am.dummy_data.get_dumbbell_heterostructure_dumbbell_lattice()
        >>> dl.refine_position_gaussian(show_progressbar=False)

        """
        if image is None:
            if self.original_image is None:
                image = self.image
            else:
                image = self.original_image
        n_tot = len(self.sublattice_list[0].atom_list)
        for i_atom in progressbar(
            range(n_tot), desc="Gaussian fitting", disable=not show_progressbar
        ):
            atom_list = []
            for sublattice in self.sublattice_list:
                atom_list.append(sublattice.atom_list[i_atom])
            afr.fit_atom_positions_gaussian(
                atom_list, image, percent_to_nn=percent_to_nn, mask_radius=mask_radius
            )

    def plot_dumbbell_distance(self, image=None, cmap=None, vmin=None, vmax=None):
        """Plot the dumbbell distances as points on an image.

        Parameters
        ----------
        image : NumPy 2D array, optional
        cmap : string
            Matplotlib colormap name, default 'viridis'
        vmin, vmax : scalars
            Min and max values for the scatter points

        Returns
        -------
        fig : matplotlib figure

        Examples
        --------
        >>> dl = am.dummy_data.get_dumbbell_heterostructure_dumbbell_lattice()
        >>> fig = dl.plot_dumbbell_distance()

        """
        if image is None:
            if self.original_image is None:
                image = self.image
            else:
                image = self.original_image
        x, y = self.dumbbell_x, self.dumbbell_y
        z = self.dumbbell_distance
        fig = pl._make_figure_scatter_point_on_image(
            image, x, y, z, cmap=cmap, vmin=vmin, vmax=vmax
        )
        return fig

    def plot_dumbbell_angle(self, image=None, cmap=None, vmin=None, vmax=None):
        """Plot the dumbbell angles as points on an image.

        Parameters
        ----------
        image : NumPy 2D array, optional
        cmap : string
            Matplotlib colormap name, default 'viridis'
        vmin, vmax : scalars
            Min and max values for the scatter points

        Returns
        -------
        fig : matplotlib figure

        Examples
        --------
        >>> dl = am.dummy_data.get_dumbbell_heterostructure_dumbbell_lattice()
        >>> fig = dl.plot_dumbbell_angle()

        """
        if image is None:
            if self.original_image is None:
                image = self.image
            else:
                image = self.original_image
        x, y = self.dumbbell_x, self.dumbbell_y
        z = self.dumbbell_angle
        fig = pl._make_figure_scatter_point_on_image(
            image, x, y, z, cmap=cmap, vmin=vmin, vmax=vmax
        )
        return fig

    def plot_dumbbell_intensity_difference(
        self, radius=4, image=None, cmap=None, vmin=None, vmax=None
    ):
        """Plot the dumbbell intensity difference as points on an image.

        Parameters
        ----------
        radius : int
            Default 4
        image : NumPy 2D array, optional
        cmap : string
            Matplotlib colormap name, default 'viridis'
        vmin, vmax : scalars
            Min and max values for the scatter points

        Returns
        -------
        fig : matplotlib figure

        Examples
        --------
        >>> dl = am.dummy_data.get_dumbbell_heterostructure_dumbbell_lattice()
        >>> fig = dl.plot_dumbbell_intensity_difference()

        See also
        --------
        get_dumbbell_intensity_difference : for getting the data itself

        """
        if image is None:
            if self.original_image is None:
                image = self.image
            else:
                image = self.original_image
        x, y = self.dumbbell_x, self.dumbbell_y
        z = self.get_dumbbell_intensity_difference(radius=radius, image=image)
        fig = pl._make_figure_scatter_point_on_image(
            image, x, y, z, cmap=cmap, vmin=vmin, vmax=vmax
        )
        return fig
