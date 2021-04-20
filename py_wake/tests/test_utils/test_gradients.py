import matplotlib.pyplot as plt
import numpy as np
from autograd import numpy as anp
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.utils.gradients import use_autograd_in, autograd, plot_gradients, fd, cs
from py_wake.tests import npt
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines import _wind_turbines
from py_wake.examples.data.hornsrev1 import V80
import pytest
from autograd.core import primitive, defvjp


@pytest.mark.parametrize('obj', [_wind_turbines, WindTurbines, V80().power, _wind_turbines.__dict__])
def test_use_autograd_in(obj):
    assert _wind_turbines.np == np
    with use_autograd_in([obj]):
        assert _wind_turbines.np == anp
    assert _wind_turbines.np == np


def test_scalar2scalar():
    def f(x):
        return x**2 + 1

    x = np.array([3])
    npt.assert_equal(cs(f)(x), 6)
    npt.assert_almost_equal(fd(f)(x), 6, 5)
    npt.assert_equal(autograd(f)(x), 6)
    pf = primitive(f)
    defvjp(pf, lambda ans, x: lambda g: g * 2 * x)
    npt.assert_array_equal(autograd(pf, False)(x), 6)


def test_vector2vector_independent():
    def f(x):
        return x**2 + 1

    def df(x):
        return 2 * x

    x = np.array([2, 3, 4])
    ref = [4, 6, 8]
    npt.assert_array_almost_equal(fd(f, False)(x), ref, 5)
    npt.assert_array_equal(cs(f, False)(x), ref)
    npt.assert_array_equal(autograd(f, False)(x), ref)

    pf = primitive(f)
    defvjp(pf, lambda ans, x: lambda g: g * df(x))
    npt.assert_array_equal(autograd(pf, False)(x), ref)


def test_vector2vector_dependent():
    def f(x):
        return x**2 + x[::-1]

    def df(x):
        return np.diag(2 * x) + np.diag(np.ones(3))[::-1]

    x = np.array([2., 3, 4])
    ref = [[4., 0., 1.],
           [0., 7., 0.],
           [1., 0., 8.]]
    npt.assert_array_almost_equal(fd(f, True)(x), ref, 5)
    npt.assert_array_almost_equal_nulp(cs(f, True)(x), ref)
    npt.assert_array_equal(autograd(f, True)(x), ref)

    pf = primitive(f)
    defvjp(pf, lambda ans, x: lambda g: np.dot(g, df(x)))
    npt.assert_array_equal(autograd(pf, True)(x), ref)


def test_multivector2vector_independent():
    def f(x, y):
        return x**2 + 2 * y**3 + 1

    def dfdx(x, y):
        return 2 * x

    def dfdy(x, y):
        return 6 * y**2

    x = np.array([2, 3, 4])
    y = np.array([1, 2, 3])
    ref_x = [4, 6, 8]
    ref_y = [6, 24, 54]
    npt.assert_array_almost_equal(fd(f, False)(x, y), ref_x, 5)
    npt.assert_array_almost_equal(fd(f, False, 1)(x, y), ref_y, 4)

    npt.assert_array_almost_equal_nulp(cs(f, False)(x, y), ref_x)
    npt.assert_array_almost_equal_nulp(cs(f, False, 1)(x, y), ref_y)

    npt.assert_array_equal(autograd(f, False)(x, y), ref_x)
    npt.assert_array_equal(autograd(f, False, 1)(x, y), ref_y)

    pf = primitive(f)
    defvjp(pf, lambda ans, x, y: lambda g: g * dfdx(x, y), lambda ans, x, y: lambda g: g * dfdy(x, y))
    npt.assert_array_equal(autograd(pf, False)(x, y), ref_x)
    npt.assert_array_equal(autograd(pf, False, 1)(x, y), ref_y)


def test_scalar2multi_scalar():
    def fxy(x):
        return x**2 + 1, 2 * x + 1

    def f(x):
        fx, fy = fxy(x)
        return fx + fy

    x = 3.
    ref = 8
    npt.assert_equal(cs(f)(x), ref)
    npt.assert_almost_equal(fd(f)(x), ref, 5)
    npt.assert_equal(autograd(f)(x), ref)

    pf = primitive(f)
    defvjp(pf, lambda ans, x: lambda g: g * (2 * x + 2))
    npt.assert_array_equal(autograd(pf, False)(x), ref)

    pf = primitive(fxy)
    defvjp(pf, lambda ans, x: lambda g: (g[0] * 2 * x, g[1] * 2))
    npt.assert_array_equal(autograd(f, False)(x), ref)


def test_vector2multi_vector():
    def fxy(x):
        return x**2 + 1, 2 * x + 1

    def f0(x):
        return fxy(x)[0]

    def fsum(x):
        fx, fy = fxy(x)
        return fx + fy

    x = np.array([1., 2, 3])
    ref0 = [2, 4, 6]
    refsum = [4, 6, 8]
    npt.assert_equal(cs(f0)(x), ref0)
    npt.assert_almost_equal(fd(f0)(x), ref0, 5)
    npt.assert_equal(autograd(f0)(x), ref0)
    pf0 = primitive(f0)
    defvjp(pf0, lambda ans, x: lambda g: g * (2 * x))
    npt.assert_array_equal(autograd(pf0, False)(x), ref0)

    npt.assert_equal(cs(fsum)(x), refsum)
    npt.assert_almost_equal(fd(fsum)(x), refsum, 5)
    npt.assert_equal(autograd(fsum)(x), refsum)
    pfsum = primitive(fsum)
    defvjp(pfsum, lambda ans, x: lambda g: g * (2 * x + 2))
    npt.assert_array_equal(autograd(pfsum, False)(x), refsum)

    pfxy = primitive(fxy)

    def dfxy(x):
        return 2 * x, np.full(x.shape, 2)

    def gsum(x):
        fx, fy = pfxy(x)
        return fx + fy

    def g0(x):
        return pfxy(x)[0]

    pgsum = primitive(gsum)
    pg0 = primitive(g0)
    defvjp(pgsum, lambda ans, x: lambda g: g * np.sum(dfxy(x), 0))
    defvjp(pg0, lambda ans, x: lambda g: g * dfxy(x)[0])

    npt.assert_array_equal(autograd(pgsum, False)(x), refsum)
    npt.assert_array_equal(autograd(pg0, False)(x), ref0)

    defvjp(pfxy, lambda ans, x: lambda g: dfxy(x)[0])

    def h0(x):
        return pfxy(x)[0]
    npt.assert_array_equal(autograd(h0, False)(x), ref0)

    defvjp(pfxy, lambda ans, x: lambda g: np.sum(g * np.asarray(dfxy(x)), 0))

    def hsum(x):
        fx, fy = pfxy(x)
        return fx + fy

    npt.assert_array_equal(autograd(hsum, False)(x), refsum)


def test_gradients():
    wt = IEA37_WindTurbines()
    wt.enable_autograd()
    ws_lst = np.arange(3, 25, .1)

    ws_pts = np.array([3., 6., 9., 12.])
    dpdu_lst = autograd(wt.power)(ws_pts)
    if 0:
        plt.plot(ws_lst, wt.power(ws_lst))
        for dpdu, ws in zip(dpdu_lst, ws_pts):
            plot_gradients(wt.power(ws), dpdu, ws, "", 1)

        plt.show()
    dpdu_ref = np.where((ws_pts > 4) & (ws_pts <= 9.8),
                        3 * 3350000 * (ws_pts - 4)**2 / (9.8 - 4)**3,
                        0)

    npt.assert_array_almost_equal(dpdu_lst, dpdu_ref)

    fd_dpdu_lst = fd(wt.power)(ws_pts)
    npt.assert_array_almost_equal(fd_dpdu_lst, dpdu_ref, 0)

    cs_dpdu_lst = cs(wt.power)(ws_pts)
    npt.assert_array_almost_equal(cs_dpdu_lst, dpdu_ref)


def test_plot_gradients():
    x = np.arange(-3, 4, .1)
    plt.plot(x, x**2)
    plot_gradients(1.5**2, 3, 1.5, "test", 1)
    if 0:
        plt.show()
    plt.close('all')
