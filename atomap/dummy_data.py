import os
import numpy as np
import hyperspy.api as hs
from hyperspy import components1d
from hyperspy.signals import EELSSpectrum
from atomap.testing_tools import MakeTestData
import atomap.atom_lattice as al
import atomap.tools as to


def _make_hexagonal_two_sublattice_testdata(image_noise=False):
    hexagon_height = 40
    dy = hexagon_height / 2
    dx = hexagon_height / 2 / 3 ** 0.5

    im_x, im_y = 300, 300

    test_data = MakeTestData(im_x, im_y)
    sigma = 3.5

    xs, ys = im_x + hexagon_height, im_y + hexagon_height

    x0, y0 = np.mgrid[dx : xs : dx * 2, 0 : ys : dy * 2]
    x0, y0 = x0.flatten(), y0.flatten()
    x1, y1 = np.mgrid[0 : xs : dx * 2, dy : ys : dy * 2]
    x1, y1 = x1.flatten(), y1.flatten()
    x_a, y_a = np.append(x0, x1), np.append(y0, y1)
    test_data.add_atom_list(x_a, y_a, sigma_x=sigma, sigma_y=sigma, amplitude=10)

    x2, y2 = np.mgrid[dx * 2 : xs : dx * 2, dy * 0.4 : ys : dy * 2]
    x2, y2 = x2.flatten(), y2.flatten()
    x3, y3 = np.mgrid[dx : xs : dx * 2, dy * 1.4 : ys : dy * 2]
    x3, y3 = x3.flatten(), y3.flatten()
    x_b, y_b = np.append(x2, x3), np.append(y2, y3)
    test_data.add_atom_list(x_b, y_b, sigma_x=sigma, sigma_y=sigma, amplitude=5)
    if image_noise:
        test_data.add_image_noise(mu=0, sigma=0.004)
    return test_data


def get_hexagonal_double_signal(image_noise=False):
    """Generate a test image signal of a hexagonal structure.

    Similar to MoS2.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.

    Returns
    -------
    >>> s = am.dummy_data.get_hexagonal_double_signal()
    >>> s.plot()

    Adding image noise

    >>> s = am.dummy_data.get_hexagonal_double_signal(image_noise=True)
    >>> s.plot()

    """
    test_data = _make_hexagonal_two_sublattice_testdata(image_noise=image_noise)
    signal = test_data.signal
    return signal


def _make_simple_cubic_with_vacancies_testdata(image_noise=False):
    test_data = MakeTestData(300, 300)
    x, y = np.mgrid[10:290:20j, 10:290:20j]
    x, y = x.flatten().tolist(), y.flatten().tolist()
    for i in [71, 193, 264]:
        x.pop(i)
        y.pop(i)
    test_data.add_atom_list(x, y, sigma_x=3, sigma_y=3)
    if image_noise:
        test_data.add_image_noise(mu=0, sigma=0.002)
    return test_data


def get_simple_cubic_with_vacancies_signal(image_noise=False):
    """Generate a test image signal with some vacancies

    Same as the simple cubic signal, but with 3 missing atoms.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.

    Returns
    -------
    signal : HyperSpy 2D

    Examples
    --------
    >>> import atomap.api as am
    >>> s = am.dummy_data.get_simple_cubic_with_vacancies_signal()
    >>> s.plot()

    With image noise

    >>> s1 = am.dummy_data.get_simple_cubic_with_vacancies_signal(
    ...     image_noise=True)
    >>> s1.plot()

    """
    test_data = _make_simple_cubic_with_vacancies_testdata(image_noise=image_noise)
    return test_data.signal


def get_simple_cubic_with_vacancies_sublattice(image_noise=False):
    """Generate a test sublattice with some vacancies

    Same as the simple cubic sublattice, but with 3 missing atoms.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.

    Returns
    -------
    signal : HyperSpy 2D

    Examples
    --------
    >>> import atomap.api as am
    >>> s = am.dummy_data.get_simple_cubic_with_vacancies_sublattice()
    >>> s.plot()

    With image noise

    >>> s1 = am.dummy_data.get_simple_cubic_with_vacancies_sublattice(
    ...     image_noise=True)
    >>> s1.plot()

    """
    test_data = _make_simple_cubic_with_vacancies_testdata(image_noise=image_noise)
    return test_data.sublattice


def _make_perovskite_001_A(image_noise=False):
    test_data = MakeTestData(252, 252)
    x, y = np.mgrid[5:252:20, 5:252:20]
    x_list = x.flatten()
    y_list = y.flatten()
    amplitude = np.ones(len(x_list)) * 20
    test_data.add_atom_list(x_list, y_list, sigma_x=3, sigma_y=3, amplitude=amplitude)
    if image_noise:
        test_data.add_image_noise(mu=0, sigma=0.004)
    return test_data


def _make_perovskite_001_B(image_noise=False):
    test_data = MakeTestData(252, 252)
    x, y = np.mgrid[15:252:20, 15:252:20]
    x_list = x.flatten()
    y_list = y.flatten()
    amplitude = np.ones(len(x_list)) * 8
    test_data.add_atom_list(x_list, y_list, sigma_x=3, sigma_y=3, amplitude=amplitude)
    if image_noise:
        test_data.add_image_noise(mu=0, sigma=0.004)
    return test_data


def get_perovskite_001_signal(image_noise=False):
    test_data_A = _make_perovskite_001_A(image_noise)
    test_data_B = _make_perovskite_001_B(image_noise)
    signal = test_data_A.signal + test_data_B.signal
    signal.metadata.General.title = "Perovskite 001 dummy data"
    signal.axes_manager[0].scale = 0.2
    signal.axes_manager[1].scale = 0.2
    signal.axes_manager[0].units = "Å"
    signal.axes_manager[1].units = "Å"
    return signal


def get_perovskite_001_atom_lattice(image_noise=False, set_element_info=True):
    test_data_A = _make_perovskite_001_A(image_noise)
    test_data_B = _make_perovskite_001_B(image_noise)
    sublattice_A = test_data_A.sublattice
    sublattice_B = test_data_B.sublattice

    image = test_data_A.signal.data + test_data_B.signal.data

    sublattice_A.image = image
    sublattice_B.image = image
    sublattice_A.original_image = image
    sublattice_B.original_image = image
    sublattice_B._plot_color = "b"
    atom_lattice = al.Atom_Lattice(
        image=image, name="Perovskite 001", sublattice_list=[sublattice_A, sublattice_B]
    )
    atom_lattice.set_scale(scale=0.2, units="Å")

    if set_element_info:
        sublattice_A.set_element_info("Sr", [0, 4, 8, 12, 16, 20, 24, 28, 32])
        sublattice_B.set_element_info("Ti", [2, 6, 10, 14, 18, 22, 26, 30])
    return atom_lattice


def _make_polarization_film_A(image_noise=False):
    test_data = MakeTestData(312, 312)
    x0, y0 = np.mgrid[5:312:20, 5:156:20]
    x0_list = x0.flatten()
    y0_list = y0.flatten()
    amplitude0 = np.ones(len(x0_list)) * 20
    test_data.add_atom_list(
        x0_list, y0_list, sigma_x=3, sigma_y=3, amplitude=amplitude0
    )
    x1, y1 = np.mgrid[5:312:20, 145 + 20 : 312 : 20]
    x1_list = x1.flatten()
    y1_list = y1.flatten()
    amplitude1 = np.ones(len(x1_list)) * 15
    test_data.add_atom_list(
        x1_list, y1_list, sigma_x=3, sigma_y=3, amplitude=amplitude1
    )
    if image_noise:
        test_data.add_image_noise(mu=0, sigma=0.004)
    return test_data


def _make_polarization_film_B(image_noise=False):
    max_x = -3
    test_data = MakeTestData(312, 312)
    x0, y0 = np.mgrid[15:312:20, 15:136:20]
    x0 = x0.astype("float64")
    y0 = y0.astype("float64")
    dx = max_x / y0.shape[1]
    for i in range(y0.shape[1]):
        x0[:, y0.shape[1] - 1 - i] += dx * i
    x0_list = x0.flatten()
    y0_list = y0.flatten()
    amplitude0 = np.ones(len(x0_list)) * 8
    test_data.add_atom_list(
        x0_list, y0_list, sigma_x=3, sigma_y=3, amplitude=amplitude0
    )
    x1, y1 = np.mgrid[15:312:20, 135 + 20 : 312 : 20]
    x1_list = x1.flatten()
    y1_list = y1.flatten()
    amplitude1 = np.ones(len(x1_list)) * 8
    test_data.add_atom_list(
        x1_list, y1_list, sigma_x=3, sigma_y=3, amplitude=amplitude1
    )
    if image_noise:
        test_data.add_image_noise(mu=0, sigma=0.004)
    return test_data


def get_polarization_film_signal(image_noise=False):
    """Get a signal with two sublattices resembling a perovskite film.

    Similar to a perovskite oxide thin film, where the B cations
    are shifted in the film.

    Parameters
    ----------
    image_noise : bool, optional
        Default False

    Returns
    -------
    signal : HyperSpy 2D signal

    Examples
    --------
    >>> import atomap.api as am
    >>> s = am.dummy_data.get_polarization_film_signal()
    >>> s.plot()

    With image noise

    >>> s = am.dummy_data.get_polarization_film_signal(image_noise=True)
    >>> s.plot()

    """
    test_data0 = _make_polarization_film_A(image_noise=image_noise)
    test_data1 = _make_polarization_film_B(image_noise=image_noise)
    signal = test_data0.signal + test_data1.signal
    return signal


def get_polarization_film_atom_lattice(image_noise=False):
    """Get an Atom Lattice with two sublattices resembling a perovskite film.

    Similar to a perovskite oxide thin film, where the B cations
    are shifted in the film.

    Parameters
    ----------
    image_noise : bool, default False

    Returns
    -------
    simple_atom_lattice : Atom_Lattice object

    Examples
    --------
    >>> import atomap.api as am
    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice()
    >>> atom_lattice.plot()

    With image noise

    >>> atom_lattice = am.dummy_data.get_polarization_film_atom_lattice(
    ...     image_noise=True)
    >>> atom_lattice.plot()

    """
    test_data0 = _make_polarization_film_A(image_noise=image_noise)
    test_data1 = _make_polarization_film_B(image_noise=image_noise)
    image = test_data0.signal.data + test_data1.signal.data

    sublattice0 = test_data0.sublattice
    sublattice1 = test_data1.sublattice
    sublattice0.image = image
    sublattice1.image = image
    sublattice0.original_image = image
    sublattice1.original_image = image
    sublattice1._plot_color = "b"
    atom_lattice = al.Atom_Lattice(
        image=image, name="Perovskite film", sublattice_list=[sublattice0, sublattice1]
    )
    return atom_lattice


def _make_simple_cubic_testdata(image_noise=False):
    simple_cubic = MakeTestData(300, 300)
    x, y = np.mgrid[10:290:20j, 10:290:20j]
    simple_cubic.add_atom_list(x.flatten(), y.flatten(), sigma_x=3, sigma_y=3)
    if image_noise:
        simple_cubic.add_image_noise(mu=0, sigma=0.002)
    return simple_cubic


def get_simple_cubic_signal(image_noise=False):
    """Generate a test image signal of a simple cubic atomic structure.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.

    Returns
    -------
    signal : HyperSpy 2D

    Examples
    --------
    >>> import atomap.api as am
    >>> s = am.dummy_data.get_simple_cubic_signal()
    >>> s.plot()

    """
    test_data = _make_simple_cubic_testdata(image_noise=image_noise)
    return test_data.signal


def get_simple_cubic_sublattice(image_noise=False):
    """Generate a test sublattice of a simple cubic atomic structure.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.

    Returns
    -------
    sublattice : Atomap Sublattice

    Examples
    --------
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_simple_cubic_sublattice()
    >>> sublattice.plot()

    """
    test_data = _make_simple_cubic_testdata(image_noise=image_noise)
    return test_data.sublattice


def _make_distorted_cubic_testdata(image_noise=False):
    test_data = MakeTestData(240, 240)
    x, y = np.mgrid[30:212:40, 30:222:20]
    x, y = x.flatten(), y.flatten()
    test_data.add_atom_list(x, y)
    x, y = np.mgrid[50:212:40, 30.0:111:20]
    x, y = x.flatten(), y.flatten()
    test_data.add_atom_list(x, y)
    x, y = np.mgrid[50:212:40, 135:222:20]
    x, y = x.flatten(), y.flatten()
    test_data.add_atom_list(x, y)
    if image_noise:
        test_data.add_image_noise(mu=0, sigma=0.002)
    return test_data


def get_distorted_cubic_signal(image_noise=False):
    """Generate a test image signal of a distorted cubic atomic structure.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.

    Returns
    -------
    signal : HyperSpy 2D

    Examples
    --------
    >>> import atomap.api as am
    >>> s = am.dummy_data.get_distorted_cubic_signal()
    >>> s.plot()

    """
    test_data = _make_distorted_cubic_testdata(image_noise=image_noise)
    return test_data.signal


def get_distorted_cubic_sublattice(image_noise=False):
    """Generate a test sublattice of a distorted cubic atomic structure.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.

    Returns
    -------
    sublattice : Atomap Sublattice

    Examples
    --------
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_distorted_cubic_sublattice()
    >>> sublattice.plot()

    """
    test_data = _make_distorted_cubic_testdata(image_noise=image_noise)
    return test_data.sublattice


def _make_atom_lists_two_sublattices(test_data_1, test_data_2=None):
    x0, y0 = np.mgrid[10:295:20, 10:300:34]
    test_data_1.add_atom_list(
        x0.flatten(), y0.flatten(), sigma_x=3, sigma_y=3, amplitude=20
    )

    if test_data_2 is None:
        test_data_2 = test_data_1

    x1, y1 = np.mgrid[10:295:20, 27:290:34]
    test_data_2.add_atom_list(
        x1.flatten(), y1.flatten(), sigma_x=3, sigma_y=3, amplitude=10
    )


def get_two_sublattice_signal():
    test_data = MakeTestData(300, 300)
    x0, y0 = np.mgrid[10:295:20, 10:300:34]
    test_data.add_atom_list(
        x0.flatten(), y0.flatten(), sigma_x=3, sigma_y=3, amplitude=20
    )

    x1, y1 = np.mgrid[10:295:20, 27:290:34]
    test_data.add_atom_list(
        x1.flatten(), y1.flatten(), sigma_x=3, sigma_y=3, amplitude=10
    )

    test_data.add_image_noise(mu=0, sigma=0.01)
    return test_data.signal


def get_simple_atom_lattice_two_sublattices(image_noise=False):
    """Returns a simple atom_lattice with two sublattices.

    Parameters
    ----------
    image_noise : bool, default False

    Returns
    -------
    simple_atom_lattice : Atom_Lattice object

    Examples
    --------
    >>> import atomap.api as am
    >>> al = am.dummy_data.get_simple_atom_lattice_two_sublattices()
    >>> al.plot()

    """
    test_data_1 = MakeTestData(300, 300)
    test_data_2 = MakeTestData(300, 300)
    test_data_3 = MakeTestData(300, 300)
    if image_noise:
        test_data_3.add_image_noise(mu=0, sigma=0.02)
    _make_atom_lists_two_sublattices(test_data_1, test_data_2)
    _make_atom_lists_two_sublattices(test_data_3)

    sublattice_1 = test_data_1.sublattice
    sublattice_2 = test_data_2.sublattice
    sublattice_2._plot_color = "b"
    image = test_data_3.signal.data
    atom_lattice = al.Atom_Lattice(
        image=image,
        name="Simple Atom Lattice",
        sublattice_list=[sublattice_1, sublattice_2],
    )
    return atom_lattice


def get_simple_heterostructure_signal(image_noise=True):
    test_data = MakeTestData(400, 400)
    x0, y0 = np.mgrid[10:390:15, 10:200:15]
    test_data.add_atom_list(
        x0.flatten(), y0.flatten(), sigma_x=3, sigma_y=3, amplitude=5
    )

    y0_max = y0.max()
    x1, y1 = np.mgrid[10:390:15, y0_max + 16 : 395 : 16]
    test_data.add_atom_list(
        x1.flatten(), y1.flatten(), sigma_x=3, sigma_y=3, amplitude=5
    )

    if image_noise:
        test_data.add_image_noise(mu=0.0, sigma=0.005)
    return test_data.signal


def get_dumbbell_signal():
    """Get a simple dumbbell signal.

    Returns
    -------
    s_dumbbell : HyperSpy Signal2D

    Example
    -------
    >>> s = am.dummy_data.get_dumbbell_signal()
    >>> s.plot()

    """
    test_data = MakeTestData(200, 200)
    x0, y0 = np.mgrid[10:200:20, 10:200:20]
    x1, y1 = np.mgrid[10:200:20, 16:200:20]
    x, y = np.vstack((x0, x1)).flatten(), np.vstack((y0, y1)).flatten()
    test_data.add_atom_list(x, y, sigma_x=1, sigma_y=1, amplitude=50)
    return test_data.signal


def get_dumbbell_heterostructure_signal():
    """Get a dumbbell heterostructure signal.

    Returns
    -------
    s_dumbbell : HyperSpy Signal2D

    Example
    -------
    >>> s = am.dummy_data.get_dumbbell_heterostructure_signal()
    >>> s.plot()

    """
    test_data = MakeTestData(500, 500)
    x0, y0 = np.mgrid[10:500:24.0, 10:250:30.0]
    x1, y1 = np.mgrid[10:500:24.0, 18:250:30.0]
    x2, y2 = np.mgrid[22:500:24.0, 25:250:30.0]
    x3, y3 = np.mgrid[22:500:24.0, 33:250:30.0]

    x0, y0 = x0.flatten(), y0.flatten()
    x1, y1 = x1.flatten(), y1.flatten()
    x2, y2 = x2.flatten(), y2.flatten()
    x3, y3 = x3.flatten(), y3.flatten()

    x0 += np.random.random(size=x0.shape) * 0.5
    y0 += np.random.random(size=y0.shape) * 0.5
    x1 += np.random.random(size=x1.shape) * 0.5
    y1 += np.random.random(size=y1.shape) * 0.5
    x2 += np.random.random(size=x2.shape) * 0.5
    y2 += np.random.random(size=y2.shape) * 0.5
    x3 += np.random.random(size=x3.shape) * 0.5
    y3 += np.random.random(size=y3.shape) * 0.5

    test_data.add_atom_list(x0, y0, sigma_x=3, sigma_y=3, amplitude=40)
    test_data.add_atom_list(x1, y1, sigma_x=3, sigma_y=3, amplitude=50)
    test_data.add_atom_list(x2, y2, sigma_x=3, sigma_y=3, amplitude=40)
    test_data.add_atom_list(x3, y3, sigma_x=3, sigma_y=3, amplitude=50)

    x4, y4 = np.mgrid[10:500:24.0, 250:500:32.0]
    x5, y5 = np.mgrid[10:500:24.0, 259:500:32.0]
    x6, y6 = np.mgrid[22:500:24.0, 266:500:32.0]
    x7, y7 = np.mgrid[22:500:24.0, 275:500:32.0]

    x4, y4 = x4.flatten(), y4.flatten()
    x5, y5 = x5.flatten(), y5.flatten()
    x6, y6 = x6.flatten(), y6.flatten()
    x7, y7 = x7.flatten(), y7.flatten()

    x4 += np.random.random(size=x4.shape) * 0.5
    y4 += np.random.random(size=y4.shape) * 0.5
    x5 += np.random.random(size=x5.shape) * 0.5
    y5 += np.random.random(size=y5.shape) * 0.5
    x6 += np.random.random(size=x6.shape) * 0.5
    y6 += np.random.random(size=y6.shape) * 0.5
    x7 += np.random.random(size=x7.shape) * 0.5
    y7 += np.random.random(size=y7.shape) * 0.5

    test_data.add_atom_list(x4, y4, sigma_x=3, sigma_y=3, amplitude=70)
    test_data.add_atom_list(x5, y5, sigma_x=3, sigma_y=3, amplitude=60)
    test_data.add_atom_list(x6, y6, sigma_x=3, sigma_y=3, amplitude=70)
    test_data.add_atom_list(x7, y7, sigma_x=3, sigma_y=3, amplitude=60)

    signal = test_data.signal
    return signal


def get_dumbbell_heterostructure_dumbbell_lattice():
    """Get a dumbbell heterostructure dumbbell lattice.

    Returns
    -------
    dumbbell_lattice : Atomap Dumbbell_Lattice

    Example
    -------
    >>> dl = am.dummy_data.get_dumbbell_heterostructure_dumbbell_lattice()
    >>> dl.plot()

    """
    test_data = MakeTestData(500, 500)
    x0, y0 = np.mgrid[10:500:24.0, 10:250:30.0]
    x1, y1 = np.mgrid[10:500:24.0, 18:250:30.0]
    x2, y2 = np.mgrid[22:500:24.0, 25:250:30.0]
    x3, y3 = np.mgrid[22:500:24.0, 33:250:30.0]

    x0, y0 = x0.flatten(), y0.flatten()
    x1, y1 = x1.flatten(), y1.flatten()
    x2, y2 = x2.flatten(), y2.flatten()
    x3, y3 = x3.flatten(), y3.flatten()

    x0 += np.random.random(size=x0.shape) * 0.5
    y0 += np.random.random(size=y0.shape) * 0.5
    x1 += np.random.random(size=x1.shape) * 0.5
    y1 += np.random.random(size=y1.shape) * 0.5
    x2 += np.random.random(size=x2.shape) * 0.5
    y2 += np.random.random(size=y2.shape) * 0.5
    x3 += np.random.random(size=x3.shape) * 0.5
    y3 += np.random.random(size=y3.shape) * 0.5

    test_data.add_atom_list(x0, y0, sigma_x=3, sigma_y=3, amplitude=40)
    test_data.add_atom_list(x1, y1, sigma_x=3, sigma_y=3, amplitude=50)
    test_data.add_atom_list(x2, y2, sigma_x=3, sigma_y=3, amplitude=40)
    test_data.add_atom_list(x3, y3, sigma_x=3, sigma_y=3, amplitude=50)

    x4, y4 = np.mgrid[10:500:24.0, 250:500:32.0]
    x5, y5 = np.mgrid[10:500:24.0, 259:500:32.0]
    x6, y6 = np.mgrid[22:500:24.0, 266:500:32.0]
    x7, y7 = np.mgrid[22:500:24.0, 275:500:32.0]

    x4, y4 = x4.flatten(), y4.flatten()
    x5, y5 = x5.flatten(), y5.flatten()
    x6, y6 = x6.flatten(), y6.flatten()
    x7, y7 = x7.flatten(), y7.flatten()

    x4 += np.random.random(size=x4.shape) * 0.5
    y4 += np.random.random(size=y4.shape) * 0.5
    x5 += np.random.random(size=x5.shape) * 0.5
    y5 += np.random.random(size=y5.shape) * 0.5
    x6 += np.random.random(size=x6.shape) * 0.5
    y6 += np.random.random(size=y6.shape) * 0.5
    x7 += np.random.random(size=x7.shape) * 0.5
    y7 += np.random.random(size=y7.shape) * 0.5

    test_data.add_atom_list(x4, y4, sigma_x=3, sigma_y=3, amplitude=70)
    test_data.add_atom_list(x5, y5, sigma_x=3, sigma_y=3, amplitude=60)
    test_data.add_atom_list(x6, y6, sigma_x=3, sigma_y=3, amplitude=70)
    test_data.add_atom_list(x7, y7, sigma_x=3, sigma_y=3, amplitude=60)

    test_data0 = MakeTestData(500, 500)
    test_data0.add_atom_list(x0, y0, sigma_x=3, sigma_y=3, amplitude=40)
    test_data0.add_atom_list(x2, y2, sigma_x=3, sigma_y=3, amplitude=40)
    test_data0.add_atom_list(x4, y4, sigma_x=3, sigma_y=3, amplitude=70)
    test_data0.add_atom_list(x6, y6, sigma_x=3, sigma_y=3, amplitude=70)
    sublattice0 = test_data0.sublattice
    sublattice0._plot_color = "blue"
    sublattice0.image = test_data.signal.data
    sublattice0.original_image = test_data.signal.data

    test_data1 = MakeTestData(500, 500)
    test_data1.add_atom_list(x1, y1, sigma_x=3, sigma_y=3, amplitude=50)
    test_data1.add_atom_list(x3, y3, sigma_x=3, sigma_y=3, amplitude=50)
    test_data1.add_atom_list(x5, y5, sigma_x=3, sigma_y=3, amplitude=60)
    test_data1.add_atom_list(x7, y7, sigma_x=3, sigma_y=3, amplitude=60)
    sublattice1 = test_data1.sublattice
    sublattice1.image = test_data.signal.data
    sublattice1.original_image = test_data.signal.data
    sublattice0.find_nearest_neighbors()
    sublattice1.find_nearest_neighbors()

    dumbbell_lattice = al.Dumbbell_Lattice(
        image=test_data.signal.data, sublattice_list=[sublattice0, sublattice1]
    )
    return dumbbell_lattice


def _add_fantasite_sublattice_A(test_data):
    xA0, yA0 = np.mgrid[10:495:15, 10:495:30]
    xA0, yA0 = xA0.flatten(), yA0.flatten()
    xA1, yA1 = xA0[0 : 8 * 17], yA0[0 : 8 * 17]
    test_data.add_atom_list(xA1, yA1, sigma_x=3, sigma_y=3, amplitude=10)
    dx = 1
    for i in range(8 * 17, 3 * 7 * 17, 2 * 17):
        xA2 = xA0[i : i + 17] + dx
        xA3 = xA0[i + 17 : i + 34] - dx
        yA2, yA3 = yA0[i : i + 17], yA0[i + 17 : i + 34]
        test_data.add_atom_list(xA2, yA2, sigma_x=3, sigma_y=3, amplitude=10)
        test_data.add_atom_list(xA3, yA3, sigma_x=3, sigma_y=3, amplitude=10)
    down = True
    for i in range(3 * 7 * 17 + 17, 580, 17):
        xA4, xA5 = xA0[i : i + 17 : 2], xA0[i + 1 : i + 17 : 2]
        if down:
            yA4 = yA0[i : i + 17 : 2] + dx
            yA5 = yA0[i + 1 : i + 17 : 2] - dx
        if not down:
            yA4 = yA0[i : i + 17 : 2] - dx
            yA5 = yA0[i + 1 : i + 17 : 2] + dx
        test_data.add_atom_list(xA4, yA4, sigma_x=3, sigma_y=3, amplitude=10)
        test_data.add_atom_list(xA5, yA5, sigma_x=3, sigma_y=3, amplitude=10)
        down = not down
    test_data.add_image_noise(mu=0, sigma=0.01, random_seed=10)
    return test_data


def _add_fantasite_sublattice_B(test_data):
    xB0, yB0 = np.mgrid[10:495:15, 25:495:30]
    xB0, yB0 = xB0.flatten(), yB0.flatten()
    test_data.add_atom_list(
        xB0[0 : 8 * 16], yB0[0 : 8 * 16], sigma_x=3, sigma_y=3, amplitude=20
    )
    xB2, yB2 = xB0[8 * 16 :], yB0[8 * 16 :]
    sig = np.arange(3, 4.1, 0.2)
    sigma_y_list = np.hstack((sig, sig[::-1], sig, sig[::-1], np.full(10, 3)))
    down = True
    for i, x in enumerate(xB2):
        rotation = 0.39
        if down:
            rotation *= -1
        sigma_y = sigma_y_list[i // 16]
        test_data.add_atom(
            x, yB2[i], sigma_x=3, sigma_y=sigma_y, amplitude=20, rotation=rotation
        )
        down = not down
    test_data.add_image_noise(mu=0, sigma=0.01, random_seed=0)
    return test_data


def _get_fantasite_atom_lattice():
    test_data1 = MakeTestData(500, 500)
    test_data1 = _add_fantasite_sublattice_A(test_data1)
    test_data2 = MakeTestData(500, 500)
    test_data2 = _add_fantasite_sublattice_B(test_data2)
    test_data3 = MakeTestData(500, 500)
    test_data3 = _add_fantasite_sublattice_A(test_data3)
    test_data3 = _add_fantasite_sublattice_B(test_data3)
    test_data3.add_image_noise(mu=0, sigma=0.01, random_seed=0)

    sublattice_1 = test_data1.sublattice
    sublattice_2 = test_data2.sublattice
    sublattice_1._plot_color = "b"
    image = test_data3.signal.data
    atom_lattice = al.Atom_Lattice(
        image=image,
        name="Fantasite Atom Lattice",
        sublattice_list=[sublattice_1, sublattice_2],
    )
    return atom_lattice


def _make_fantasite_test_data():
    test_data = MakeTestData(500, 500)
    test_data = _add_fantasite_sublattice_A(test_data)
    test_data = _add_fantasite_sublattice_B(test_data)
    test_data.add_image_noise(mu=0, sigma=0.01, random_seed=0)
    return test_data


def get_fantasite():
    """
    Fantasite is a fantastic structure with several interesting structural
    variations.

    It contains two sublattices, domains with elliptical atomic
    columns and tilt-patterns. This function returns a HyperSpy 2D signal.

    Examples
    --------
    >>> import atomap.api as am
    >>> s = am.dummy_data.get_fantasite()
    >>> s.plot()

    Returns
    -------
    fantasite_signal : HyperSpy 2D signal

    See also
    --------
    get_fantasite_sublattice : get a sublattice object of the fantasite.

    """
    test_data = _make_fantasite_test_data()
    return test_data.signal


def get_fantasite_sublattice():
    """
    Fantasite is a fantastic structure with several interesting structural
    variations.

    It contains two sublattices, domains with elliptical atomic
    columns and tilt-patterns. This function returns an Atomap sublattice.

    Currently this function returns the two sublattices as one sublattice.
    To get these sublattices separately, see get_fantasite_atom_lattice

    Examples
    --------
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_fantasite_sublattice()
    >>> sublattice.plot()

    See also
    --------
    get_fantasite : get a Signal2D object of the fantasite.
    get_fantasite_atom_lattice : get the atom lattice
        for fantasite, with both sublattices.

    """
    test_data = _make_fantasite_test_data()
    return test_data.sublattice


def get_fantasite_atom_lattice():
    """
    Fantasite is a fantastic structure with several interesting structural
    variations.

    It contains two sublattices, domains with elliptical atomic
    columns and tilt-patterns. This function returns an Atomap Atom_Lattice.

    Examples
    --------
    >>> import atomap.api as am
    >>> atom_lattice = am.dummy_data.get_fantasite_atom_lattice()
    >>> atom_lattice.plot()

    """
    atom_lattice = _get_fantasite_atom_lattice()
    return atom_lattice


def get_perovskite110_ABF_signal(image_noise=False):
    """Returns signals representing an ABF image of a perovskite <110>.

    Parameters
    ----------
    image_noise : default False
        If True, will add Gaussian noise to the image.

    Returns
    -------
    signal : HyperSpy 2D,

    Examples
    --------
    >>> import atomap.api as am
    >>> s_ABF = am.dummy_data.get_perovskite110_ABF_signal()

    """
    test_data = MakeTestData(300, 300)
    x_A, y_A = np.mgrid[10:295:20, 10:300:34]
    test_data.add_atom_list(
        x_A.flatten(), y_A.flatten(), sigma_x=5, sigma_y=5, amplitude=40
    )
    x_B, y_B = np.mgrid[10:295:20, 27:290:34]
    test_data.add_atom_list(
        x_B.flatten(), y_B.flatten(), sigma_x=3, sigma_y=3, amplitude=10
    )
    x_O, y_O = np.mgrid[0:295:20, 27:290:34]
    test_data.add_atom_list(
        x_O.flatten(), y_O.flatten(), sigma_x=2, sigma_y=2, amplitude=1.2
    )

    if image_noise:
        test_data.add_image_noise(mu=0, sigma=0.01)

    ABF = 1 - test_data.signal.data
    s_ABF = hs.signals.Signal2D(ABF)
    return s_ABF


def _make_eels_map_spatial_image_la(x_size=100, y_size=100):
    test_data_la = MakeTestData(x_size, y_size)
    la_x, la_y = np.mgrid[5:100:20, 5:100:10]
    la_x, la_y = la_x.flatten(), la_y.flatten()
    test_data_la.add_atom_list(la_x, la_y, amplitude=20, sigma_x=2.5, sigma_y=2.5)
    la_spatial = test_data_la.signal
    return la_spatial


def _make_eels_map_spatial_image_mn(x_size=100, y_size=100):
    test_data_mn = MakeTestData(x_size, y_size)
    mn_x, mn_y = np.mgrid[15:100:20, 5:100:10]
    mn_x, mn_y = mn_x.flatten(), mn_y.flatten()
    test_data_mn.add_atom_list(mn_x, mn_y, amplitude=5, sigma_x=2, sigma_y=2)
    mn_spatial = test_data_mn.signal
    return mn_spatial


def _make_mn_eels_spectrum(energy_range=None):
    if energy_range is None:
        energy_range = (590, 900)
    energy = np.arange(energy_range[0], energy_range[1], 1)
    mn_arctan = components1d.Arctan(A=2, k=0.2, x0=634)
    mn_arctan.minimum_at_zero = True
    mn_l3_g = components1d.Gaussian(A=100, centre=642, sigma=1.8)
    mn_l2_g = components1d.Gaussian(A=40, centre=652, sigma=1.8)

    mn_data = mn_arctan.function(energy)
    mn_data += mn_l3_g.function(energy)
    mn_data += mn_l2_g.function(energy)
    return mn_data


def _make_la_eels_spectrum(energy_range=None):
    if energy_range is None:
        energy_range = (590, 900)
    energy = np.arange(energy_range[0], energy_range[1], 1)
    la_arctan = components1d.Arctan(A=1, k=0.2, x0=845)
    la_arctan.minimum_at_zero = True
    la_l3_g = components1d.Gaussian(A=110, centre=833, sigma=1.5)
    la_l2_g = components1d.Gaussian(A=100, centre=850, sigma=1.5)

    la_data = la_arctan.function(energy)
    la_data += la_l3_g.function(energy)
    la_data += la_l2_g.function(energy)
    return la_data


def get_eels_spectrum_survey_image():
    """Get an artificial survey image of atomic resolution EELS map of LaMnO3

    Returns
    -------
    survey_image : HyperSpy Signal2D

    Example
    -------
    >>> import atomap.api as am
    >>> s = am.dummy_data.get_eels_spectrum_survey_image()
    >>> s.plot()

    See also
    --------
    get_eels_spectrum_map : corresponding EELS map

    """
    s = _make_eels_map_spatial_image_la() + _make_eels_map_spatial_image_mn()
    s = s.swap_axes(0, 1)
    return s


def get_eels_spectrum_map(add_noise=True):
    """Get an artificial atomic resolution EELS map of LaMnO3

    Containing the Mn-L23 and La-M54 edges.

    Parameters
    ----------
    add_noise : bool
        If True, will add Gaussian noise to the spectra.
        Default True.

    Returns
    -------
    eels_map : HyperSpy EELSSpectrum

    Example
    -------
    >>> import atomap.api as am
    >>> s_eels_map = am.dummy_data.get_eels_spectrum_map()
    >>> s_eels_map.plot()

    Not adding noise

    >>> s_eels_map = am.dummy_data.get_eels_spectrum_map(add_noise=False)

    See also
    --------
    get_eels_spectrum_survey_image : signal with same spatial dimensions

    """
    x_size, y_size = 100, 100
    e0, e1 = 590, 900

    la_spatial = _make_eels_map_spatial_image_la(x_size=x_size, y_size=y_size)
    mn_spatial = _make_eels_map_spatial_image_mn(x_size=x_size, y_size=y_size)

    # Generate EELS spectra
    mn_data = _make_mn_eels_spectrum(energy_range=(e0, e1))
    la_data = _make_la_eels_spectrum(energy_range=(e0, e1))

    # Generate 3D-data
    # La
    data_3d_la = np.zeros(shape=(x_size, y_size, (e1 - e0)))
    data_3d_la[:, :] = la_data
    temp_3d_la = np.zeros(shape=(x_size, y_size, (e1 - e0)))
    temp_3d_la = temp_3d_la.swapaxes(0, 2)
    temp_3d_la[:] += la_spatial.data
    temp_3d_la = temp_3d_la.swapaxes(0, 2)
    data_3d_la *= temp_3d_la

    # Mn
    data_3d_mn = np.zeros(shape=(x_size, y_size, (e1 - e0)))
    data_3d_mn[:, :] = mn_data
    temp_3d_mn = np.zeros(shape=(x_size, y_size, (e1 - e0)))
    temp_3d_mn = temp_3d_mn.swapaxes(0, 2)
    temp_3d_mn[:] += mn_spatial.data
    temp_3d_mn = temp_3d_mn.swapaxes(0, 2)
    data_3d_mn *= temp_3d_mn

    data_3d = data_3d_mn + data_3d_la

    # Adding background and add noise
    background = components1d.PowerLaw(A=1e10, r=3, origin=0)
    background_data = background.function(np.arange(e0, e1, 1))
    temp_background_data = np.zeros(shape=(x_size, y_size, (e1 - e0)))
    temp_background_data[:, :] += background_data
    data_3d += background_data

    if add_noise:
        data_noise = np.random.random((x_size, y_size, (e1 - e0))) * 0.7
        data_3d += data_noise

    s_3d = EELSSpectrum(data_3d)
    return s_3d


def get_single_atom_sublattice():
    """Get a sublattice containing a single atom.

    Returns
    -------
    sublattice_single_atom : Atomap Sublattice class

    Example
    -------
    >>> sublattice = am.dummy_data.get_single_atom_sublattice()
    >>> sublattice.plot()

    """
    test_data = MakeTestData(50, 50)
    test_data.add_atom(25, 20, 2, 2)
    sublattice = test_data.sublattice
    return sublattice


def get_precipitate_signal():
    """Get a signal emulating a preciptate embedded in a matrix.

    Returns
    -------
    signal_percipitate : HyperSpy Signal2D

    Examples
    --------
    >>> s = am.dummy_data.get_precipitate_signal()
    >>> s.plot()

    """
    verts = [[250, 100], [100, 250], [250, 400], [400, 250]]

    test_data = MakeTestData(500, 500)

    x0, y0 = np.mgrid[0:505:20.0, 0:505:20.0]
    x0, y0 = x0.flatten(), y0.flatten()
    positions0 = np.stack((x0, y0)).T
    positions0_selected = to._get_atom_selection_from_verts(
        positions0, verts, invert_selection=True
    )
    x0, y0 = positions0_selected.T
    x0 += np.random.random(size=x0.shape)
    y0 += np.random.random(size=y0.shape)
    test_data.add_atom_list(x0, y0, sigma_x=3, sigma_y=3)

    x1, y1 = np.mgrid[2.5:505:15.0, 2.5:505:15.0]
    x1, y1 = x1.flatten(), y1.flatten()
    x1, y1 = to._rotate_points_around_position(250, 250, x1, y1, 45)
    positions1 = np.stack((x1, y1)).T
    positions1_selected = to._get_atom_selection_from_verts(
        positions1, verts, invert_selection=False
    )
    x1, y1 = positions1_selected.T
    x1 += np.random.random(size=x1.shape)
    y1 += np.random.random(size=y1.shape)
    test_data.add_atom_list(x1, y1, sigma_x=3, sigma_y=3, amplitude=2)

    s = test_data.signal
    return s


def get_atom_counting_signal():
    """Get an artificial image of 4 different atomic column intensities.

    Returns
    -------
    s_atom_counting : HyperSpy 2D signal

    Examples
    --------
    >>> import atomap.api as am
    >>> s = am.dummy_data.get_atom_counting_signal()
    >>> s.plot()

    """
    testdata = MakeTestData(200, 200)
    for i in range(4):
        x, y = np.mgrid[60 * i : (i + 1) * 60 : 15, 10:200:15]
        x, y = x.flatten(), y.flatten()
        testdata.add_atom_list(
            x, y, sigma_x=2, sigma_y=2, amplitude=(i + 1) * 20, rotation=0.4
        )
    testdata.add_image_noise(sigma=0.02)
    return testdata.signal


def get_scanning_distortion_sublattice():
    """Get an artificial sublattice emulating scanning distortions.

    Returns
    -------
    sublattice : Atomap Sublattice

    Example
    -------
    >>> import atomap.api as am
    >>> sublattice = am.dummy_data.get_scanning_distortion_sublattice()
    >>> sublattice.plot()

    """
    test_data = MakeTestData(500, 500, add_row_scan_distortion=1)
    x, y = np.mgrid[25:475:20j, 25:475:20j]
    x, y = x.flatten(), y.flatten()
    test_data.add_atom_list(x, y, sigma_x=4, sigma_y=4)
    sublattice = test_data.sublattice
    return sublattice


def get_nanoparticle_signal():
    """Get an simulated nanoparticle image.

    Returns
    -------
    signal : HyperSpy Signal2D

    Example
    -------
    >>> import atomap.api as am
    >>> s = am.dummy_data.get_nanoparticle_signal()
    >>> s.plot()

    """
    my_path = os.path.dirname(__file__)
    path = os.path.join(my_path, "example_data", "simulated_nanoparticle.hspy")
    s = hs.load(path)
    return s
