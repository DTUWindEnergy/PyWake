
import matplotlib.pyplot as plt
from py_wake import np
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.utils import gradients
from py_wake.utils.gradients import fd, cs, autograd, plot_gradients, hypot, cabs, interp, set_gradient_function
from xarray.core.dataset import Dataset
#
#
# @pytest.mark.parametrize('obj', [_wind_turbines, WindTurbines, V80().power, _wind_turbines.__dict__])
# def test_use_autograd_in(obj):
#     _wind_turbines.np = np
#     assert _wind_turbines.np == np
#     with AutogradNumpy():
#         assert _wind_turbines.np.abs == anp.abs  # @UndefinedVariable
#     assert _wind_turbines.np == np

from numpy import testing as npt

from jax import custom_vjp
import pytest


def test_scalar2scalar():
    def f(x):
        return x**2 + 1

    x = 3.

    npt.assert_equal(cs(f)(x), 6)
    npt.assert_almost_equal(fd(f)(x), 6, 5)
    npt.assert_equal(autograd(f)(x), np.array(6))
    assert autograd(f)(x).dtype == np.float64

    pf = custom_vjp(f)
    # wrong gradient, to check that function is actually used
    pf.defvjp(lambda x: (f(x), (x,)), lambda res, g: (g * 2 * res[0] + 1,))
    npt.assert_array_equal(autograd(pf)(x), 7)

    cf = set_gradient_function(lambda x: 2 * x + 2)(f)
    npt.assert_array_equal(autograd(cf)(x), 8)


def test_vector2vector_independent():
    def f(x):
        return np.array(x**2 + 1)

    def df(x):
        return 2 * x + 1

    x = np.array([2., 3, 4])
    ref = np.array([4, 6, 8])
    npt.assert_array_almost_equal(fd(f, False)(x), ref, 5)
    npt.assert_array_equal(cs(f, False)(x), ref)
    npt.assert_array_equal(autograd(f, False)(x), ref)

    pf = custom_vjp(f)
    pf.defvjp(lambda x: (f(x), (df(x),)), lambda res, g: (g * res[0],))
    npt.assert_array_equal(autograd(pf, False)(x), ref + 1)


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

    pf = custom_vjp(f)
    pf.defvjp(lambda x: (f(x), (df(x),)), lambda res, g: (np.dot(g, res[0]),))
    npt.assert_array_equal(autograd(pf, True)(x), ref)


def test_multivector2vector_independent():
    def f(x, y):
        return x**2 + 2 * y**3 + 1

    def dfdx(x, y):
        return 2 * x

    def dfdy(x, y):
        return 6 * y**2

    x = np.array([2., 3, 4])
    y = np.array([1., 2, 3])
    ref_x = np.array([4, 6, 8])
    ref_y = np.array([6, 24, 54])
    npt.assert_array_almost_equal(fd(f, False)(x, y), ref_x, 5)
    npt.assert_array_almost_equal(fd(f, False, 1)(x, y), ref_y, 4)

    npt.assert_array_almost_equal_nulp(cs(f, False)(x, y), ref_x)
    npt.assert_array_almost_equal_nulp(cs(f, False, 1)(x, y), ref_y)

    npt.assert_array_equal(autograd(f, False)(x, y), ref_x)
    npt.assert_array_equal(autograd(f, False, 1)(x, y), ref_y)

    pf = custom_vjp(f)
    pf.defvjp(lambda x, y: (f(x, y), (dfdx(x, y), dfdy(x, y))), lambda res, g: (g * res[0], g * res[1]))
    npt.assert_array_equal(autograd(pf, False)(x, y), ref_x)
    npt.assert_array_equal(autograd(pf, False, 1)(x, y), ref_y)


def test_scalar2multi_scalar():
    def fxy(x):
        return x**2 + 1, 2 * x + 1

    def f(x):
        fx, fy = fxy(x)
        return fx + fy

    x = 3.
    ref = np.array(8)
    npt.assert_equal(cs(f)(x), ref)
    npt.assert_almost_equal(fd(f)(x), ref, 5)
    npt.assert_equal(autograd(f)(x), ref)

    pf = custom_vjp(f)
    pf.defvjp(lambda x: (f(x), (x,)), lambda res, g: (2 * res[0] + 2,))
    npt.assert_array_equal(autograd(pf, False)(x), ref)

    pf = custom_vjp(fxy)
    pf.defvjp(lambda x: (fxy(x), (x,)), lambda res, g: (g[0] * 2 * res[0], g[1] * 2))
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
    ref0 = np.array([2, 4, 6])
    refsum = np.array([4, 6, 8])
    npt.assert_equal(cs(f0, False)(x), ref0)
    npt.assert_almost_equal(fd(f0, False)(x), ref0, 5)
    npt.assert_array_equal(autograd(f0, False)(x), ref0)
    pf0 = custom_vjp(f0)
    pf0.defvjp(lambda x: (f0(x), (x,)), lambda res, g: (g * (2 * x),))
    npt.assert_array_equal(autograd(pf0, False)(x), ref0)

    npt.assert_equal(cs(fsum, False)(x), refsum)
    npt.assert_almost_equal(fd(fsum, False)(x), refsum, 5)
    npt.assert_array_equal(autograd(fsum, False)(x), refsum)
    pfsum = custom_vjp(fsum)
    pfsum.defvjp(lambda x: (fsum(x), (x,)), lambda res, g: (g * (2 * x + 2),))
    npt.assert_array_equal(autograd(pfsum, False)(x), refsum)

    pfxy = custom_vjp(fxy)

    def dfxy(x):
        return np.array([2 * x, np.full(x.shape, 2)])

    def gsum(x):
        fx, fy = fxy(x)
        return fx + fy

    def g0(x):
        return pfxy(x)[0]

    pgsum = custom_vjp(gsum)
    pg0 = custom_vjp(g0)
    pgsum.defvjp(lambda x: (pgsum(x), (x,)),
                 lambda res, g: (g * np.sum(dfxy(res[0]), 0),))
    pg0.defvjp(lambda x: (pgsum(x), (x,)), lambda res, g: (g * dfxy(res[0])[0],))

    npt.assert_array_equal(autograd(pgsum, False)(x), refsum)
    npt.assert_array_equal(autograd(pg0, False)(x), ref0)

    pfxy.defvjp(
        lambda x: (pfxy(x), (x,)),
        lambda res, g: (dfxy(res[0])[0],))

    def h0(x):
        return np.array(pfxy(x)[0])
    npt.assert_array_equal(autograd(h0, False)(x), ref0)

    pfxy.defvjp(lambda x: (fxy(x), (x,)),
                lambda res, g: (np.sum(np.asarray(g) * np.asarray(dfxy(res[0])), 0),))

    def hsum(x):
        fx, fy = pfxy(x)
        return fx + fy

    npt.assert_array_equal(autograd(hsum, False)(x), refsum)


def test_wrt_2d():
    def f(x):
        return np.sum(x**2)

    for grad in [fd, cs, autograd]:
        for x in [2., np.arange(4), np.arange(6).reshape((3, 2))]:
            assert np.shape(grad(f)(np.array(x, dtype=np.float64))) == np.shape(x)


def test_2d_wrt_2d():
    def f2d(x):
        return np.reshape(np.sum(x**2) * np.arange(6), (2, 3))

    for grad in [fd, cs, autograd]:
        for x in [2, np.arange(3), np.arange(20).reshape((4, 5))]:
            assert np.shape(grad(f2d)(np.array(x, dtype=np.float64))) == (2, 3) + np.shape(x)


def test_autograd_wrt_xy():
    def f(x, y, z):
        return x**2 + 2 * y**3 + z

    def dfdx(x, y):
        return 2 * x

    def dfdy(x, y):
        return 6 * y**2

    x = np.array([2., 3, 4])
    y = np.array([1., 2, 3])
    ref_x = np.array([4, 6, 8])
    ref_y = np.array([6, 24, 54])

    dfdxy = autograd(f, vector_interdependence=False, argnum=[0, 1])(x, y=y, z=1)

    npt.assert_array_equal(dfdxy, np.array([ref_x, ref_y]))


def test_gradients():
    wt = IEA37_WindTurbines()
    ws_lst = np.arange(3, 25, .1)

    ws_pts = np.array([3., 6., 9., 12.])
    dpdu_lst = autograd(wt.power, False)(ws_pts)
    if 0:
        plt.plot(ws_lst, wt.power(ws_lst))
        for dpdu, ws in zip(dpdu_lst.tolist(), ws_pts):
            plot_gradients(wt.power(ws), dpdu, ws, "", 1)

        plt.show()
    dpdu_ref = np.where((ws_pts > 4) & (ws_pts <= 9.8),
                        3 * 3350000 * (ws_pts - 4)**2 / (9.8 - 4)**3,
                        0)

    npt.assert_array_almost_equal(dpdu_lst, dpdu_ref)

    fd_dpdu_lst = fd(wt.power, False)(ws_pts)
    npt.assert_array_almost_equal(fd_dpdu_lst, dpdu_ref, 0)

    cs_dpdu_lst = cs(wt.power, False)(ws_pts)
    npt.assert_array_almost_equal(cs_dpdu_lst, dpdu_ref)


def test_plot_gradients():
    x = np.arange(-3, 4, .1)
    plt.plot(x, x**2)
    plot_gradients(1.5**2, 3, 1.5, "test", 1)
    if 0:
        plt.show()
    plt.close('all')


def test_hypot():
    # Test real.
    a = np.array([3, 9])
    b = np.array([4, 40])
    npt.assert_equal(hypot(a, b), np.array([5, 41]))
    npt.assert_array_almost_equal(cs(hypot)(a, b), autograd(hypot)(a, b))
    # Test complex.
    a = 3 + 4j
    b = 1 - 2j
    npt.assert_array_almost_equal(hypot(a, b), 2.486028939392892 + 4.022479320953552j)


def test_cabs():
    a = [-5, 6]
    npt.assert_array_equal(cabs(a), np.abs(a))
    npt.assert_array_almost_equal(fd(cabs, False)(a), [-1, 1], 10)
    npt.assert_array_equal(cs(cabs, False)(a), [-1, 1])
    npt.assert_array_equal(autograd(cabs, False)(a), [-1, 1])


def test_arctan2():
    for x in [-.5, 0, .5]:
        for y in [-.4, 0, .4]:
            npt.assert_array_almost_equal(gradients.arctan2(y + 0j, x).real, gradients.arctan2(y, x), 15)
            dydx_lst = [grad(gradients.arctan2)(y, x) for grad in [fd, cs, autograd]]
            if x != 0 and y != 0:
                npt.assert_array_almost_equal(dydx_lst[0], dydx_lst[1])
            if not (x == 0 and y == 0):
                npt.assert_array_almost_equal(dydx_lst[1], dydx_lst[2])


def test_gradients_interp():
    xp, x, y = np.array([5, 16]), np.array([0, 10, 20]), np.array([100, 200, 400])

    def f(xp):
        return 2 * gradients.interp(xp, x=x, y=y)
    npt.assert_array_equal(interp(xp, x, y), np.interp(xp, x, y))
    npt.assert_array_almost_equal(fd(f, False)(xp), [20, 40])
    npt.assert_array_equal(cs(f, False)(xp), [20, 40])
    npt.assert_array_equal(autograd(f, False)(xp), [20, 40])


def test_gradients_logaddexp():

    x = [0, 0, 0, 1]
    y = [1, 100, 1000, 1]

    def f(x, y):
        return 2 * gradients.logaddexp(x, y)

    dfdx = 2 * (np.exp(x - np.logaddexp(x, y)))
    dfdy = 2 * (np.exp(y - np.logaddexp(x, y)))
    npt.assert_array_equal(f(x, y), 2 * np.logaddexp(x, y))
    npt.assert_array_almost_equal(fd(f, False)(x, y), dfdx)
    npt.assert_array_almost_equal(fd(f, False, argnum=1)(x, y), dfdy)
    npt.assert_array_almost_equal(cs(f, False)(x, y), dfdx)
    npt.assert_array_almost_equal(cs(f, False, argnum=1)(x, y), dfdy)
    npt.assert_array_equal(autograd(f, False)(x, y), dfdx)
    npt.assert_array_equal(autograd(f, False, argnum=1)(x, y), dfdy)


def test_set_gradient_function():
    def df(x):
        return 3 * x

    @set_gradient_function(df)
    def f(x):
        return x**2

    assert f(4) == 16
    npt.assert_almost_equal(fd(f, False)(4), 8, 5)
    npt.assert_almost_equal(cs(f, False)(4), 8)
    npt.assert_array_equal(autograd(f, False)([4, 5]), [12, 15])


def test_set_gradient_function_cls():
    class T():
        def df(self, x):
            return 3 * x

        @set_gradient_function(df)
        def f(self, x):
            return x**2

    t = T()
    assert t.f(4) == 16
    npt.assert_almost_equal(fd(t.f)(4), 8, 5)
    npt.assert_almost_equal(cs(t.f)(4), 8)
    assert autograd(t.f)(4) == 12


def test_set_gradient_function_kwargs():
    def df(x):
        return 3 * x

    @set_gradient_function(df)
    def f(x):
        return x**2

    def f2(x):
        return f(x)

    assert f2(x=4) == 16
    npt.assert_almost_equal(fd(f2)(x=4), 8, 5)
    npt.assert_almost_equal(cs(f2)(x=4), 8)
    assert autograd(f2)(x=4) == 12


@pytest.mark.parametrize('interpolator', [gradients.PchipInterpolator, gradients.UnivariateSpline])
def test_Interpolators(interpolator):
    xp = np.linspace(0, 2 * np.pi, 100000)
    x = np.linspace(0, 2 * np.pi, 10)
    y = np.sin(x)
    plt.plot(xp, np.sin(xp), label='sin(x)')
    plt.plot(x, np.sin(x), '.')

    interp = interpolator(x, y)
    x = x[3]
    dfdx_lst = [method(interp)(x) for method in [fd, cs, autograd]]
    npt.assert_array_almost_equal(dfdx_lst[0], dfdx_lst[1])
    npt.assert_array_equal(dfdx_lst[1], dfdx_lst[2])
    gradients.color_dict = {}
    if 0:
        plt.plot(xp, interp(xp), label='Interpolated')
        dfdx = np.cos(x)
        gradients.plot_gradients(np.sin(x), dfdx, x, label='analytical', step=.5)

        for method, dfdx in zip([fd, cs, autograd], dfdx_lst):
            gradients.plot_gradients(interp(x), dfdx, x, label=method.__name__, step=1)
        plt.legend()
        plt.show()


@pytest.mark.parametrize('y,x,axis', [([2, 3, 7, 9], [1, 2, 4, 8], 0),
                                      (lambda x:-x**2 + 9, np.linspace(-3, 3, 10), 0)])
def test_trapz(y, x, axis):
    if callable(y):
        y = y(x)

    npt.assert_array_equal(np.trapz(y, x, axis), gradients.trapz(y, x, axis))
    dtrapz_dy_lst = [method(gradients.trapz, True)(y, x, axis) for method in [fd, cs, autograd]]
    npt.assert_array_almost_equal(dtrapz_dy_lst[0], dtrapz_dy_lst[1])
    npt.assert_array_equal(dtrapz_dy_lst[1], dtrapz_dy_lst[2])

    if x is not None:
        dtrapz_dx_lst = [method(gradients.trapz, True, argnum=1)(y, x, axis) for method in [fd, cs, autograd]]
        npt.assert_array_almost_equal(dtrapz_dx_lst[0], dtrapz_dx_lst[1])
        npt.assert_array_almost_equal(dtrapz_dx_lst[1], dtrapz_dx_lst[2], 14)


@pytest.mark.parametrize('test', [
    lambda y, x: gradients.trapz(np.reshape(y, (2, 4)), np.reshape(x, (2, 4)), axis=1),
    lambda y, x: gradients.trapz(np.reshape(y, (2, 4)).T, np.reshape(x, (2, 4)).T, axis=0),
    lambda y, x: gradients.trapz(np.reshape(y, (1, 2, 4, 1)), np.reshape(x, (1, 2, 4, 1)), axis=2)
])
def test_trapz_axis(test):
    y, x = np.array([2, 3, 7, 9] * 2), np.array([1, 2, 4, 8] * 2)

    autograd(test, True, argnum=1)(y, x)
    autograd(test, True)(y, x)
    dtrapz_dy_lst = [method(test, True)(y, x) for method in [fd, cs, autograd]]
    npt.assert_array_almost_equal(dtrapz_dy_lst[0], dtrapz_dy_lst[1])
    npt.assert_array_equal(dtrapz_dy_lst[1], dtrapz_dy_lst[2])

    if x is not None:

        dtrapz_dx_lst = [method(test, True, argnum=1)(y, x) for method in [fd, cs, autograd]]
        npt.assert_array_almost_equal(dtrapz_dx_lst[0], dtrapz_dx_lst[1])
        npt.assert_array_almost_equal(dtrapz_dx_lst[1], dtrapz_dx_lst[2], 14)


def test_multiple_inputs():
    def f(x, y):
        return x * y

    for method in [fd, cs, autograd]:
        npt.assert_array_almost_equal(method(f, False, argnum=[0, 1])(np.array([2, 3, 4]), np.array([4, 5, 6])),
                                      [[4, 5, 6], [2, 3, 4]], 8)


# def test_asarray_xarray():
#     def f(x):
#         ds = Dataset({'x': ('i', x)}, coords={'i': range(len(x))})
#         str(ds.x)  # calls asarray
#         return ds.x.values * 2
#
#     autograd(f)([2, 3, 4])


def test_erf():
    x = .5
    dfdx = [method(gradients.erf, True)(x) for method in [fd, cs, autograd]]
    npt.assert_array_almost_equal(dfdx[0], dfdx)


def test_mod():
    dfdx = [method(gradients.mod, True)(7., 3.) for method in [fd, cs, autograd]]
    npt.assert_array_almost_equal(dfdx[0], dfdx)


def test_modf():
    x = 7.3
    f_ref = np.modf(x)
    npt.assert_array_almost_equal(f_ref, gradients.modf(x))

    dfdx_ref = (np.array(np.modf(x + 1e-6)) - np.array(np.modf(x))) / 1e-6
    for method in [fd, cs, autograd]:
        dfdx = method(gradients.modf, True)(7.3)
        npt.assert_array_almost_equal(dfdx_ref, dfdx)


def test_gamma():
    x = .5
    dfdx = [method(gradients.gamma, True)(x) for method in [fd, cs, autograd]]
    npt.assert_array_almost_equal(dfdx[0], dfdx, 5)


def test_sqrt():
    x = [4, 0]

    def f(x):
        return np.sqrt(x)

    dfdx = np.array([method(f, False)(x) for method in [fd, cs, autograd]])

    npt.assert_array_almost_equal(np.array(dfdx)[:, 0][0], .25, 5)
    assert np.isinf()
