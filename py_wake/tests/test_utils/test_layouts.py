import matplotlib.pyplot as plt
from py_wake.utils.layouts import rectangle, square, circular, farm_area
from py_wake.examples.data.hornsrev1 import V80
from py_wake.examples.data.iea37._iea37 import IEA37Site
from py_wake.tests import npt
import pytest
import numpy as np


def test_square():
    wt = V80()
    x, y = square(25, wt.diameter() * 5)
    if 0:
        wt.plot(x, y)
        plt.show()
    assert len(x) == 25
    assert x[-1] == y[-1] == 80 * 5 * (5 - 1)


def test_rectangle():
    wt = V80()
    x, y = rectangle(8, 5, wt.diameter() * 5)
    if 0:
        wt.plot(x, y)
        plt.show()
    assert len(x) == 8
    assert x[-1] == 800
    assert y[-1] == 400


def test_circular():
    site = IEA37Site(64)
    npt.assert_array_almost_equal(circular([1, 5, 12, 18, 28], 3000), site.initial_position.T, 4)


@pytest.mark.parametrize('xy,area', [(square(25, 100), 400 * 400),
                                     (rectangle(20, 5, 100), 400 * 300),
                                     (circular([1, 5, 12, 18, 100], 3000), np.pi * 3000**2)])
def test_area(xy, area):
    if 0:
        plt.plot(*xy, '.')
        plt.show()
    npt.assert_allclose(farm_area(*xy), area, rtol=0.001)
