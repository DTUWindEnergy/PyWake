from py_wake.utils.eq_distance_interpolator import EqDistRegGridInterpolator
import pytest
import numpy as np
import matplotlib.pyplot as plt
from py_wake.tests import npt
import xarray as xr
from py_wake.tests.check_speed import timeit


def test_eqdistance_interpolator_not_equal():
    with pytest.raises(ValueError, match="Axes must be equidistant"):
        EqDistRegGridInterpolator([[10, 21, 30]], [1, 2, 3])


def test_eqdistance_interpolator_not_match():
    with pytest.raises(ValueError, match="Lengths of x does not match shape of V"):
        EqDistRegGridInterpolator([[5, 10, 15], [200, 300]], np.arange(12).reshape(3, 4))


def test_eqdistance_interpolator_wrong_method():
    with pytest.raises(ValueError, match='Method must be "linear" or "nearest"'):
        EqDistRegGridInterpolator([[10, 20, 30]], [1, 2, 3], method="cubic")(None)


def test_eqdistance_interpolator_outside_area():
    with pytest.raises(ValueError, match='Outside data area'):
        EqDistRegGridInterpolator([[10, 20, 30]], [1, 2, 3])(9.9)
    with pytest.raises(ValueError, match='Outside data area'):
        EqDistRegGridInterpolator([[10, 20, 30]], [1, 2, 3])(30.1)
    with pytest.raises(ValueError, match="Outside data area"):
        EqDistRegGridInterpolator([[5, 10, 15], [200, 300]], np.arange(6).reshape(3, 2))([(5, 300.1)])


def test_eqdistance_interpolator_2d():
    x = [5, 10, 15]
    y = [200, 300, 400, 500]
    v = np.arange(12).reshape(3, 4)
    eq = EqDistRegGridInterpolator([x, y], v)
    xp = [[i, 200] for i in np.arange(5, 16)]
    npt.assert_array_almost_equal(eq(xp), np.linspace(0, 8, 11))
    npt.assert_array_almost_equal(eq(xp, 'nearest'), np.round(np.linspace(0, 8, 11) / 4) * 4)

    X, Y = np.meshgrid(x, y)
    xp, yp = np.linspace(5, 15, 11), np.linspace(200, 500, 13)
    Xp, Yp = np.meshgrid(xp, yp)

    co = np.array([Xp, Yp]).T.reshape(-1, 2)
    for method in ['nearest', 'linear']:
        Z = eq(co, method).reshape(Xp.T.shape)

        plt.figure()
        c = plt.contourf(Xp, Yp, Z.T, np.arange(12))
        plt.plot(X, Y, 'xw',)
        plt.colorbar(c)
    if 0:
        plt.show()
    plt.close()


def test_eqdistance_interpolator_2d_plus_1d():
    eq = EqDistRegGridInterpolator([[5, 10, 15], [200, 300, 400, 500]], np.arange(12).repeat(2).reshape(3, 4, 2))
    x = [[i, 200] for i in np.arange(5, 16)]
    npt.assert_array_almost_equal(eq(x).sum(0), np.linspace(0, 8, 11) * 2)


def compare_speed():
    x = np.arange(0, 100)
    y = np.arange(10, 100)
    z = np.arange(30, 33)
    wd = np.arange(360)
    ws = np.arange(3, 26)
    V = np.random.rand(len(x), len(y), len(z), len(wd), len(ws))
    print(V.shape)
    da = xr.DataArray(V, coords=[('x', x), ('y', y), ('z', z), ('wd', wd), ('ws', ws)])
    coords = xr.Dataset(coords={'wd': np.arange(360), 'ws': [9, 10], 'i': np.arange(16),
                                'x': ('i', np.linspace(0, 9, 16)),
                                'y': ('i', np.linspace(10, 20, 16)),
                                'z': ('i', np.linspace(30, 32, 16))})

    x1, t = timeit(da.sel_interp_all, verbose=True)(coords)

    I, L, K = len(coords.x), len(coords.wd), len(coords.ws)

    def interp():
        c = np.array([coords.x.data.repeat(L * K), coords.y.data.repeat(L * K), coords.z.data.repeat(L * K),
                      np.tile(coords.wd.data.repeat(K), I), np.tile(coords.ws.data, I * L)]).T
        return EqDistRegGridInterpolator([x, y, z, wd, ws], da.values)(c)
    x2, t = timeit(interp, verbose=True)()
    npt.assert_array_almost_equal(x1, x2.reshape(16, len(coords.wd), len(coords.ws)))
