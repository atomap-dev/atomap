from pytest import approx
import numpy as np
import atomap.fitting_tools as ft


class TestODRFitter:
    def test_simple(self):
        x = np.arange(0, 10, 1)
        y = np.arange(0, 20, 2)
        beta = ft.ODR_linear_fitter(x, y)
        assert approx(beta[0], abs=1e-7) == 2
        assert approx(beta[1], abs=1e-7) == 0

    def test_flat(self):
        x = np.arange(0, 10, 1)
        y = np.full_like(x, 1)
        beta = ft.ODR_linear_fitter(x, y)
        assert approx(beta[0], abs=1e-7) == 0
        assert approx(beta[1], abs=1e-7) == 1

    def test_vertical(self):
        y = np.arange(0, 10, 1)
        x = np.full_like(y, 5)
        beta = ft.ODR_linear_fitter(x, y)
        y_vector = [0, ft.linear_fit_func(beta, 5)]
        x_vector = [1, 0]
        dot = np.dot(x_vector, y_vector)
        assert dot == 0.0


class TestFindDistancePointLine:
    def test_simple(self):
        x_list, y_list = [0], [0]
        line = [0, 1]
        d = ft.get_shortest_distance_point_to_line(x_list, y_list, line)
        assert d[0] == 1.0

    def test_negative(self):
        x_list, y_list = [0], [0]
        line = [0, -1]
        d = ft.get_shortest_distance_point_to_line(x_list, y_list, line)
        assert d == -1.0

    def test_vertical_line(self):
        x_list, y_list = [0, 2], [1, 1]
        line = [400000, -400000]
        d = ft.get_shortest_distance_point_to_line(x_list, y_list, line)
        assert approx(d[0], rel=1e-5) == -1.0
        assert approx(d[1], rel=1e-5) == 1.0

    def test_60_degreees(self):
        x_list, y_list = [0], [0]
        line = [-np.tan(np.radians(30)), 3]
        d = ft.get_shortest_distance_point_to_line(x_list, y_list, line)
        assert approx(d[0], rel=1e-4) == 2.598

    def test_45_degreees(self):
        x_list, y_list = [0], [0]
        line = [1, -3]
        d = ft.get_shortest_distance_point_to_line(x_list, y_list, line)
        assert approx(d[0], rel=1e-12) == -3 * np.sin(np.radians(45))

    def test_on_line(self):
        x_list, y_list = [1], [1]
        line = [1, 0]
        d = ft.get_shortest_distance_point_to_line(x_list, y_list, line)
        assert approx(d[0], abs=1e-12) == 0
