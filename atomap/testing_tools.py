import math
import numpy as np
from numpy.random import normal
from hyperspy.misc.utils import isiterable
from atomap.sublattice import Sublattice
from atomap.atom_position import Atom_Position
from atomap.atom_lattice import Atom_Lattice


class MakeTestData(object):
    def __init__(
        self,
        image_x,
        image_y,
        sublattice_generate_image=True,
        sigma_quantile=5,
        add_row_scan_distortion=None,
        show_progressbar=False,
    ):
        """Class for generating test datasets of atomic resolution STEM images.

        Parameters
        ----------
        image_x, image_y : int
            Size of the image data.
        sublattice_generate_image : bool, default True
            When generating sublattices, a raster image is generated to
            complement the atom position objects (found in sublattice.image).
            For large amounts of atom positions, this can take a very long
            time. If sublattice_generate_image is False, this image will not
            be generated. Useful for generating sublattice objects for testing
            quicker, when only the atom positions themselves are needed.
        sigma_quantile : scalar
            When generating the image of Gaussians
            (if sublattice_generate_image is True),
            only a subset of the region around each atom is calculated.
            This region is defined by the (x, y) position of the atom, and the
            maximum sigma (max(sx, sy)) * sigma_quantile. Setting a high sigma
            will mean the Gaussians are calculated further out, but this leads
            to the image generating process being slower. Default 5.
        add_row_scan_distortion : scalar
            Shift horizontal rows to emulate scanning distortions.
            By default this is off, set to a number to enable.
            The amount of distortion is proportional to the number given.
        show_progressbar : bool
            If True, will show a progressbar when generating the image data,
            which is normally the most time consuming part.
            Default False.

        Attributes
        ----------
        signal : HyperSpy 2D Signal
        sublattice : Atomap Sublattice
        atom_lattice : Atomap Atom_Lattice
        gaussian_list : list of 2D Gaussians objects

        Examples
        --------
        >>> from atomap.testing_tools import MakeTestData
        >>> test_data = MakeTestData(200, 200)
        >>> test_data.add_atom(x=10, y=20)
        >>> test_data.signal.plot()

        Adding many atoms

        >>> test_data = MakeTestData(200, 200)
        >>> import numpy as np
        >>> x, y = np.mgrid[0:200:10j, 0:200:10j]
        >>> x, y = x.flatten(), y.flatten()
        >>> test_data.add_atom_list(x, y)
        >>> test_data.signal.plot()

        Adding many atoms with different parameters

        >>> test_data = MakeTestData(200, 200)
        >>> x, y = np.mgrid[0:200:10j, 0:200:10j]
        >>> x, y = x.flatten(), y.flatten()
        >>> sx, sy = np.random.random(len(x)), np.random.random(len(x))
        >>> A, r = np.random.random(len(x))*10, np.random.random(len(x))*3.14
        >>> test_data.add_atom_list(x, y, sigma_x=sx, sigma_y=sy,
        ...         amplitude=A, rotation=r)
        >>> test_data.signal.plot()

        The class also generates a sublattice object

        >>> test_data = MakeTestData(200, 200)
        >>> import numpy as np
        >>> x, y = np.mgrid[0:200:10j, 0:200:10j]
        >>> x, y = x.flatten(), y.flatten()
        >>> test_data.add_atom_list(x, y)
        >>> test_data.sublattice.plot()

        Also Atom_Lattice objects

        >>> atom_lattice = test_data.atom_lattice
        >>> atom_lattice.plot()

        Adding scanning distortions

        >>> test_data = MakeTestData(500, 500, add_row_scan_distortion=1)
        >>> x, y = np.mgrid[25:475:20j, 25:475:20j]
        >>> x, y = x.flatten(), y.flatten()
        >>> test_data.add_atom_list(x, y, sigma_x=5, sigma_y=5)
        >>> test_data.signal.plot()

        Generating a sublattice with 22500 atoms quickly, by not
        generating the image

        >>> test_data = MakeTestData(200, 200, sublattice_generate_image=False)
        >>> import numpy as np
        >>> x, y = np.mgrid[0:1000:150j, 0:1000:150j]
        >>> x, y = x.flatten(), y.flatten()
        >>> test_data.add_atom_list(x, y)
        >>> sublattice = test_data.sublattice

        """
        self.data_extent = (image_x, image_y)
        self._image_noise = False
        self._sublattice_generate_image = sublattice_generate_image
        self.__sublattice = Sublattice([], np.zeros((2, 2)))
        self.__sublattice.atom_list = []
        self._sigma_quantile = sigma_quantile
        self._add_row_scan_distortion = add_row_scan_distortion
        self._show_progressbar = show_progressbar

    @property
    def signal(self):
        signal = self.__sublattice.get_model_image(
            image_shape=self.data_extent,
            sigma_quantile=self._sigma_quantile,
            show_progressbar=self._show_progressbar,
        )
        if self._image_noise is not False:
            signal.data += self._image_noise
        if self._add_row_scan_distortion is not None:
            self._shift_vertical_rows(signal)
        return signal

    @property
    def gaussian_list(self):
        gaussian_list = []
        for atom in self.__sublattice.atom_list:
            gaussian_list.append(atom.as_gaussian())
        return gaussian_list

    @property
    def sublattice(self):
        atom_list = []
        for atom in self.__sublattice.atom_list:
            new_atom = Atom_Position(
                x=atom.pixel_x,
                y=atom.pixel_y,
                sigma_x=atom.sigma_x,
                sigma_y=atom.sigma_y,
                rotation=atom.rotation,
                amplitude=atom.amplitude_gaussian,
            )
            atom_list.append(new_atom)

        if self._sublattice_generate_image:
            image = self.signal.data
        else:
            image = np.zeros(self.data_extent[::-1])
        sublattice = Sublattice([], image)
        sublattice.atom_list = atom_list
        return sublattice

    @property
    def atom_lattice(self):
        sublattice = self.sublattice
        atom_lattice = Atom_Lattice(
            image=sublattice.image, sublattice_list=[sublattice]
        )
        return atom_lattice

    def _shift_vertical_rows(self, signal):
        scale = self._add_row_scan_distortion
        shift_list = np.random.standard_normal(self.data_extent[1]) * scale
        shift_list = np.round(shift_list).astype(np.int)
        for iy, shift in enumerate(shift_list):
            signal.data[iy] = np.roll(signal.data[iy], shift)

    def add_atom(self, x, y, sigma_x=1, sigma_y=1, amplitude=1, rotation=0):
        """
        Add a single atom to the test data.

        Parameters
        ----------
        x, y : numbers
            Position of the atom.
        sigma_x, sigma_y : numbers, default 1
        amplitude : number, default 1
        rotation : number, default 0

        Examples
        --------
        >>> from atomap.testing_tools import MakeTestData
        >>> test_data = MakeTestData(200, 200)
        >>> test_data.add_atom(x=10, y=20)
        >>> test_data.signal.plot()
        """
        atom = Atom_Position(
            x=x,
            y=y,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            rotation=rotation,
            amplitude=amplitude,
        )
        self.__sublattice.atom_list.append(atom)

    def add_atom_list(self, x, y, sigma_x=1, sigma_y=1, amplitude=1, rotation=0):
        """
        Add several atoms to the test data.

        Parameters
        ----------
        x, y : iterable
            Position of the atoms. Must be iterable, and have the same size.
        sigma_x, sigma_y : number or iterable, default 1
            If number: all the atoms will have the same sigma.
            Use iterable for setting different sigmas for different atoms.
            If iterable: must be same length as x and y iterables.
        amplitude : number or iterable, default 1
            If number: all the atoms will have the same amplitude.
            Use iterable for setting different amplitude for different atoms.
            If iterable: must be same length as x and y iterables.
        rotation : number or iterable, default 0
            If number: all the atoms will have the same rotation.
            Use iterable for setting different rotation for different atoms.
            If iterable: must be same length as x and y iterables.

        Examples
        --------
        >>> from atomap.testing_tools import MakeTestData
        >>> test_data = MakeTestData(200, 200)
        >>> import numpy as np
        >>> x, y = np.mgrid[0:200:10j, 0:200:10j]
        >>> x, y = x.flatten(), y.flatten()
        >>> test_data.add_atom_list(x, y)
        >>> test_data.signal.plot()

        """
        if len(x) != len(y):
            raise ValueError("x and y needs to have the same length")

        if isiterable(sigma_x):
            if len(sigma_x) != len(x):
                raise ValueError("sigma_x and x needs to have the same length")
        else:
            sigma_x = [sigma_x] * len(x)

        if isiterable(sigma_y):
            if len(sigma_y) != len(y):
                raise ValueError("sigma_y and x needs to have the same length")
        else:
            sigma_y = [sigma_y] * len(x)

        if isiterable(amplitude):
            if len(amplitude) != len(x):
                raise ValueError("amplitude and x needs to have the same length")
        else:
            amplitude = [amplitude] * len(x)

        if isiterable(rotation):
            if len(rotation) != len(x):
                raise ValueError("rotation and x needs to have the same length")
        else:
            rotation = [rotation] * len(x)
        iterator = zip(x, y, sigma_x, sigma_y, amplitude, rotation)
        for tx, ty, tsigma_x, tsigma_y, tamplitude, trotation in iterator:
            self.add_atom(tx, ty, tsigma_x, tsigma_y, tamplitude, trotation)

    def add_image_noise(self, mu=0, sigma=0.005, only_positive=False, random_seed=None):
        """
        Add white noise to the image signal. The noise component is Gaussian
        distributed, with a default expectation value at 0, and a sigma of
        0.005. If only_positive is set to True, the absolute value of the
        noise is added to the signal. This can be useful for avoiding negative
        values in the image signal.

        Parameters
        ----------
        mu : int, float
            The expectation value of the Gaussian distribution, default is 0
        sigma : int, float
            The standard deviation of the Gaussian distribution, default
            is 0.005.
        only_positive : bool
            Default is False. If True, the absolute value of the noise is added
            to the image signal.
        random_seed : int, optional
            Set the random seed of the noise, which gives the same image
            noise each time. Useful for testing and comparing images.

        Example
        -------
        >>> from atomap.testing_tools import MakeTestData
        >>> test_data = MakeTestData(300, 300)
        >>> import numpy as np
        >>> x, y = np.mgrid[10:290:15j, 10:290:15j]
        >>> test_data.add_atom_list(x.flatten(), y.flatten(), sigma_x=3,
        ...     sigma_y=3)
        >>> test_data.add_image_noise()
        >>> test_data.signal.plot()

        Using a specific random seed

        >>> test_data.add_image_noise(random_seed=0)

        """
        if random_seed is not None:
            np.random.seed(random_seed)
        shape = self.signal.axes_manager.shape
        noise = normal(mu, sigma, shape)
        if only_positive:
            self._image_noise = np.absolute(noise)
        else:
            self._image_noise = noise


def make_vector_test_gaussian(x, y, standard_deviation=1, n=30):
    point_list = []
    for i in range(n):
        g_x = normal(x, scale=standard_deviation)
        g_y = normal(y, scale=standard_deviation)
        point_list.append([g_x, g_y])
    point_list = np.array(point_list)
    return point_list


def make_nn_test_dataset(xN=3, yN=3, xS=9, yS=9, std=0.3, n=50):
    point_list = np.array([[], []]).T
    for ix in range(-xN, xN + 1):
        for iy in range(-yN, yN + 1):
            if (ix == 0) and (iy == 0):
                pass
            else:
                gaussian_list = make_vector_test_gaussian(
                    ix * xS, iy * yS, standard_deviation=std, n=n
                )
                point_list = np.vstack((point_list, gaussian_list))
    return point_list


def find_atom_position_match(component_list, atom_list, delta=3, scale=1.0):
    match_list = []
    for atom in atom_list:
        for component in component_list:
            x = atom.pixel_x * scale - component.centre_x.value
            y = atom.pixel_y * scale - component.centre_y.value
            d = math.hypot(x, y)
            if d < delta:
                match_list.append([component, atom])
                break
    return match_list


def get_fit_miss_array(match_list):
    fit_miss = []
    for match in match_list:
        x = match[0].centre_x.value - match[1].pixel_x
        y = match[0].centre_y.value - match[1].pixel_y
        d = math.hypot(x, y)
        fit_miss.append([x, y, d])
    fit_miss = np.array(fit_miss)
    return fit_miss


class _DummyEvent:
    def __init__(self):
        self.key = None


def _do_move_press_release_event(fig, poly, x, y):
    ax = fig.axes[0]
    xd, yd = fig.axes[0].transData.transform((x, y))
    move_event = _DummyEvent()
    move_event.x, move_event.y = xd, yd
    move_event.xdata, move_event.ydata = x, y
    move_event.inaxes = ax
    poly.onmove(move_event)

    press_event = _DummyEvent()
    press_event.x, press_event.y = xd, yd
    press_event.xdata, press_event.ydata = x, y
    press_event.inaxes = ax
    press_event.button = 1
    poly.press(press_event)

    release_event = _DummyEvent()
    release_event.x, release_event.y = xd, yd
    release_event.xdata, release_event.ydata = x, y
    release_event.inaxes = ax
    release_event.button = 1

    poly.release(release_event)


def _do_several_move_press_release_event(fig, poly, position_list):
    for x, y in position_list:
        _do_move_press_release_event(fig, poly, x, y)
