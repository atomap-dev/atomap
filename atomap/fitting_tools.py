from math import sqrt
import numpy as np
import scipy.odr


def linear_fit_func(p, t):
    return p[0] * t + p[1]


def ODR_linear_fitter(x, y):
    """Orthogonal Distance Regression linear fitting.

    A least squares fitting method with perpendicular offsets. Often called
    Orthogonal Distance Regression (ODR). This method works better than
    Ordinary Least Squares fitting (OLS, with vertical offsets), on vertical
    lines.

    Parameters
    ----------
    x : array of x-values
    y : array of y-values

    Returns
    -------
    beta : array
        Array with the coefficients for y = ax + b.
        a = beta[0], b = beta[1]

    Example
    --------
    >>> import atomap.fitting_tools as ft
    >>> x, y = [0, 1, 2, 3], [1, 0, -1, -2]
    >>> ft.ODR_linear_fitter(x, y)
    array([-1.,  1.])

    """
    Model = scipy.odr.Model(linear_fit_func)
    Data = scipy.odr.RealData(x, y)
    Odr = scipy.odr.ODR(Data, Model, [10000, 1], maxit=10000)
    output = Odr.run()
    beta = output.beta
    return beta


def get_shortest_distance_point_to_line(x_list, y_list, line):
    """Calculates the shortest distance from each point in a list to a line.

    Parameters
    ----------
    x_list, y_list : list of numbers
    line : list
        The line is defined by y = ax + b, given by in the form [a, b].

    Returns
    -------
    list of numbers : list of the distance from the points to the line

    Examples
    -------
    Horizontal line, and points horizontal

    >>> import atomap.fitting_tools as ft
    >>> x_list, y_list = [0, 1, 2, 3], [1, 1, 1, 1]
    >>> ft.get_shortest_distance_point_to_line(x_list, y_list, [0, 0])
    array([-1., -1., -1., -1.])

    Horizontal line, and points vertical

    >>> x_list, y_list = [1, 1, 1, 1], [0, 1, 2, 3]
    >>> ft.get_shortest_distance_point_to_line(x_list, y_list, [0, 0])
    array([ 0., -1., -2., -3.])

    """
    x0, y0 = np.asarray(x_list), np.asarray(y_list)
    a, b, c = line[0], -1, line[1]
    num = a * x0 + b * y0 + c
    den = sqrt(a ** 2 + b ** 2)
    d = num / den
    return d
